"""Benchmark tests for the I/O operations."""

from pathlib import Path

import pytest

import polars as pl

pytestmark = pytest.mark.benchmark()


def test_write_read_scan_large_csv(groupby_data: pl.DataFrame, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    data_path = tmp_path / "data.csv"
    groupby_data.write_csv(data_path)

    predicate = pl.col("v2") < 5

    shape_eager = pl.read_csv(data_path).filter(predicate).shape
    shape_lazy = pl.scan_csv(data_path).filter(predicate).collect().shape

    assert shape_lazy == shape_eager


@pytest.fixture(scope="module")
def adaptive_predicate_parquet(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, int]:
    num_rows = 500_000
    df = pl.DataFrame(
        {
            "seed_a": pl.int_range(0, num_rows, eager=True) % 10 == 0,
            "seed_b": pl.int_range(0, num_rows, eager=True) % 100 < 10,
            "expensive": ["needle-" + "x" * 64, "other-" + "y" * 64] * (num_rows // 2),
            "value": pl.int_range(0, num_rows, eager=True),
        }
    )
    path = tmp_path_factory.mktemp("adaptive-predicate") / "data.parquet"
    df.write_parquet(path, row_group_size=5_000)
    predicate = (
        pl.col("seed_a")
        & pl.col("seed_b")
        & pl.col("expensive").str.contains("^needle")
    )
    expected = df.filter(predicate)["value"].sum()
    assert expected is not None
    return path, expected


@pytest.mark.parametrize("adaptive", [False, True], ids=["legacy", "adaptive"])
def test_scan_parquet_adaptive_predicates(
    adaptive_predicate_parquet: tuple[Path, int],
    adaptive: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, expected = adaptive_predicate_parquet
    if adaptive:
        monkeypatch.setenv("POLARS_PQ_ADAPTIVE_PREDICATE_DECODE", "1")
    else:
        monkeypatch.delenv("POLARS_PQ_ADAPTIVE_PREDICATE_DECODE", raising=False)

    predicate = (
        pl.col("seed_a")
        & pl.col("seed_b")
        & pl.col("expensive").str.contains("^needle")
    )
    result = (
        pl.scan_parquet(path, parallel="prefiltered")
        .filter(predicate)
        .select(pl.col("value").sum())
        .collect(engine="streaming")
        .item()
    )
    assert result == expected
