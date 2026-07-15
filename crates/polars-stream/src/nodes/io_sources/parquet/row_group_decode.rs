use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use polars_async::executor::TaskPriority;
use polars_async::primitives::opt_spawned_future::parallelize_first_to_local;
use polars_core::frame::DataFrame;
use polars_core::prelude::{ArrowField, BooleanChunked, ChunkFilter, Column, DataType, IntoColumn};
use polars_core::series::Series;
use polars_core::utils::arrow::bitmap::{Bitmap, BitmapBuilder, MutableBitmap};
use polars_error::PolarsResult;
use polars_io::RowIndex;
use polars_io::predicates::{
    ColumnPredicateExpr, ColumnPredicates, ScanIOPredicate, SpecializedColumnPredicate,
};
pub use polars_io::prelude::_internal::PrefilterMaskSetting;
use polars_io::prelude::try_set_sorted_flag;
use polars_parquet::read::{Filter, PredicateFilter, PrimitiveLogicalType};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::{IdxSize, UnitVec};

use super::row_group_data_fetch::RowGroupData;
use crate::nodes::io_sources::parquet::projection::ArrowFieldProjection;

const ADAPTIVE_TRAINING_SAMPLES: u32 = 4;
const ADAPTIVE_EWMA_ALPHA: f64 = 0.25;
const ADAPTIVE_MIN_IMPROVEMENT: f64 = 0.8;
const ADAPTIVE_MAX_PREDICTED_SEED_DENSITY: f64 = 0.25;
const ADAPTIVE_MAX_ACTUAL_SEED_DENSITY: f64 = 0.5;

#[derive(Clone, Copy, Debug, Default)]
struct PredicateObservation {
    eager_cost_per_row: f64,
    eager_selectivity: f64,
    eager_samples: u32,
    masked_cost_per_row: f64,
    masked_selectivity: f64,
    masked_samples: u32,
}

impl PredicateObservation {
    fn update_eager(&mut self, cost_per_row: f64, selectivity: f64) {
        update_ewma(
            &mut self.eager_cost_per_row,
            self.eager_samples,
            cost_per_row,
        );
        update_ewma(&mut self.eager_selectivity, self.eager_samples, selectivity);
        self.eager_samples = self.eager_samples.saturating_add(1);
    }

    fn update_masked(&mut self, cost_per_row: f64, selectivity: f64) {
        update_ewma(
            &mut self.masked_cost_per_row,
            self.masked_samples,
            cost_per_row,
        );
        update_ewma(
            &mut self.masked_selectivity,
            self.masked_samples,
            selectivity,
        );
        self.masked_samples = self.masked_samples.saturating_add(1);
    }
}

fn update_ewma(current: &mut f64, samples: u32, sample: f64) {
    if samples == 0 {
        *current = sample;
    } else {
        *current = ADAPTIVE_EWMA_ALPHA * sample + (1.0 - ADAPTIVE_EWMA_ALPHA) * *current;
    }
}

pub(super) struct AdaptivePredicateState {
    observations: Mutex<Vec<PredicateObservation>>,
    verbose: bool,
}

impl AdaptivePredicateState {
    pub(super) fn new(num_predicates: usize, verbose: bool) -> Self {
        Self {
            observations: Mutex::new(vec![PredicateObservation::default(); num_predicates]),
            verbose,
        }
    }

    fn record_eager(&self, predicate_idx: usize, elapsed_ns: f64, rows: usize, passed: usize) {
        if rows == 0 {
            return;
        }
        let mut observations = self.observations.lock();
        observations[predicate_idx].update_eager(
            elapsed_ns / rows as f64,
            (passed as f64 / rows as f64).clamp(0.0, 1.0),
        );
    }

    fn record_masked(&self, predicate_idx: usize, elapsed_ns: f64, rows: usize, passed: usize) {
        if rows == 0 {
            return;
        }
        self.observations.lock()[predicate_idx].update_masked(
            elapsed_ns / rows as f64,
            (passed as f64 / rows as f64).clamp(0.0, 1.0),
        );
    }

    fn choose_seed(&self, masked_cost_factors: &[f64]) -> Option<Vec<usize>> {
        let observations = self.observations.lock();
        debug_assert_eq!(observations.len(), masked_cost_factors.len());
        if observations.len() < 2
            || observations
                .iter()
                .any(|o| o.eager_samples < ADAPTIVE_TRAINING_SAMPLES)
        {
            return None;
        }

        let legacy_cost = observations
            .iter()
            .map(|o| o.eager_cost_per_row)
            .fold(0.0, f64::max);
        if legacy_cost == 0.0 {
            return None;
        }

        let mut order = (0..observations.len()).collect::<Vec<_>>();
        order.sort_by(|&a, &b| {
            let score = |o: PredicateObservation| {
                o.eager_cost_per_row / (1.0 - o.eager_selectivity).max(0.01)
            };
            score(observations[a])
                .total_cmp(&score(observations[b]))
                .then_with(|| a.cmp(&b))
        });

        let mut best: Option<(f64, Vec<usize>)> = None;
        let mut seed_density = 1.0;
        let mut seed_cost: f64 = 0.0;

        for seed_len in 1..order.len() {
            let predicate_idx = order[seed_len - 1];
            let observation = observations[predicate_idx];
            seed_density *= observation.eager_selectivity;
            seed_cost = seed_cost.max(observation.eager_cost_per_row);

            if seed_density > ADAPTIVE_MAX_PREDICTED_SEED_DENSITY {
                continue;
            }

            let masked_cost = order[seed_len..]
                .iter()
                .map(|&idx| {
                    let observation = observations[idx];
                    if observation.masked_samples > 0 {
                        observation.masked_cost_per_row
                    } else {
                        observation.eager_cost_per_row * masked_cost_factors[idx]
                    }
                })
                .fold(0.0, f64::max);
            let staged_cost = seed_cost + seed_density * masked_cost;

            if staged_cost <= legacy_cost * ADAPTIVE_MIN_IMPROVEMENT
                && best
                    .as_ref()
                    .is_none_or(|(best_cost, _)| staged_cost < *best_cost)
            {
                best = Some((staged_cost, order[..seed_len].to_vec()));
            }
        }

        if self.verbose
            && let Some((staged_cost, seed)) = best.as_ref()
        {
            eprintln!(
                "[ParquetFileReader]: Adaptive predicate plan {:?}, predicted cost: {:.3} / {:.3}",
                seed, staged_cost, legacy_cost
            );
        }

        best.map(|(_, seed)| seed)
    }
}

/// Turns row group data into DataFrames.
pub(super) struct RowGroupDecoder {
    pub(super) num_pipelines: usize,
    pub(super) projected_arrow_fields: Arc<[ArrowFieldProjection]>,
    pub(super) allow_column_predicates: bool,
    pub(super) row_index: Option<RowIndex>,
    pub(super) predicate: Option<ScanIOPredicate>,
    pub(super) use_prefiltered: Option<PrefilterMaskSetting>,
    /// Indices into `projected_arrow_fields. This must be sorted.
    pub(super) predicate_field_indices: Arc<[usize]>,
    /// Indices into `projected_arrow_fields. This must be sorted.
    pub(super) non_predicate_field_indices: Arc<[usize]>,
    pub(super) target_values_per_thread: usize,
    pub(super) adaptive_predicate_state: Option<Arc<AdaptivePredicateState>>,
}

impl RowGroupDecoder {
    pub(super) async fn row_group_data_to_df(
        &self,
        mut row_group_data: RowGroupData,
    ) -> PolarsResult<DataFrame> {
        // If the slice consumes the entire row-group. Don't slice. This allows for prefiltering to
        // happen more often until we properly support prefiltering with pre-slices.
        row_group_data.slice.take_if(|slice| {
            slice.0 == 0 && slice.1 >= row_group_data.row_group_metadata.num_rows()
        });

        if self.use_prefiltered.is_some()
            && row_group_data.slice.is_none()
            && !self.predicate_field_indices.is_empty()
        {
            self.row_group_data_to_df_prefiltered(row_group_data).await
        } else {
            self.row_group_data_to_df_impl(row_group_data).await
        }
    }

    async fn row_group_data_to_df_impl(
        &self,
        row_group_data: RowGroupData,
    ) -> PolarsResult<DataFrame> {
        let row_group_data = Arc::new(row_group_data);

        let out_width = self.row_index.is_some() as usize + self.projected_arrow_fields.len();

        let mut out_columns = Vec::with_capacity(out_width);

        let slice_range = row_group_data
            .slice
            .map(|(offset, len)| offset..offset + len)
            .unwrap_or(0..row_group_data.row_group_metadata.num_rows());

        assert!(slice_range.end <= row_group_data.row_group_metadata.num_rows());

        if let Some(s) = self.materialize_row_index(row_group_data.as_ref(), slice_range.clone())? {
            out_columns.push(s);
        }

        let mut decoded_cols = Vec::with_capacity(row_group_data.row_group_metadata.n_columns());
        self.decode_projected_columns(
            &mut decoded_cols,
            &row_group_data,
            Some(polars_parquet::read::Filter::Range(slice_range.clone())),
        )
        .await?;

        drop(row_group_data);

        let projection_height = slice_range.len();

        out_columns.extend(decoded_cols);

        let df = unsafe { DataFrame::new_unchecked(projection_height, out_columns) };

        let df = if let Some(predicate) = self.predicate.as_ref() {
            let mask = predicate.predicate.evaluate_io(&df)?;
            let mask = mask.bool().unwrap();

            let filtered =
                filter_cols(df.into_columns(), mask, self.target_values_per_thread).await?;

            let height = if let Some(fst) = filtered.first() {
                fst.len()
            } else {
                mask.num_trues()
            };

            unsafe { DataFrame::new_unchecked(height, filtered) }
        } else {
            df
        };

        assert_eq!(df.width(), out_width); // `out_width` should have been calculated correctly

        Ok(df)
    }

    fn materialize_row_index(
        &self,
        row_group_data: &RowGroupData,
        slice_range: core::ops::Range<usize>,
    ) -> PolarsResult<Option<Column>> {
        if let Some(RowIndex { name, offset }) = self.row_index.clone() {
            let projection_height = slice_range.len();

            let offset = offset.saturating_add(
                IdxSize::try_from(row_group_data.row_offset + slice_range.start)
                    .unwrap_or(IdxSize::MAX),
            );

            // The DataFrame can be empty at this point if no columns were projected from the file,
            // so we create the row index column manually instead of using `df.with_row_index` to
            // ensure it has the correct number of rows.
            Ok(Some(Column::new_row_index(
                name,
                offset,
                projection_height,
            )?))
        } else {
            Ok(None)
        }
    }

    /// Potentially parallelizes based on number of rows & columns. Decoded columns are appended to
    /// `out_vec`.
    async fn decode_projected_columns(
        &self,
        out_vec: &mut Vec<Column>,
        row_group_data: &Arc<RowGroupData>,
        filter: Option<polars_parquet::read::Filter>,
    ) -> PolarsResult<()> {
        let projected_arrow_fields = &self.projected_arrow_fields;
        let expected_num_rows = filter
            .as_ref()
            .map_or(row_group_data.row_group_metadata.num_rows(), |x| {
                x.num_rows(row_group_data.row_group_metadata.num_rows())
            });

        // Ensure we provide the same output column order as the pre-filtered decode.
        let get_projected_field_at_output_index = {
            let predicate_field_indices = self.predicate_field_indices.clone();
            let non_predicate_field_indices = self.non_predicate_field_indices.clone();

            move |i: usize| {
                if predicate_field_indices.is_empty() {
                    i
                } else if i < predicate_field_indices.len() {
                    predicate_field_indices[i]
                } else {
                    non_predicate_field_indices[i - predicate_field_indices.len()]
                }
            }
        };

        let cols_per_thread = calc_cols_per_thread(
            row_group_data.row_group_metadata.num_rows(),
            self.target_values_per_thread,
        );

        let projected_arrow_fields = projected_arrow_fields.clone();
        let row_group_data_2 = row_group_data.clone();

        let task_handles = {
            let projected_arrow_fields = projected_arrow_fields.clone();
            let filter = filter.clone();

            parallelize_first_to_local(
                TaskPriority::Low,
                (0..projected_arrow_fields.len())
                    .step_by(cols_per_thread)
                    .map(move |offset| {
                        let row_group_data = row_group_data_2.clone();
                        let projected_arrow_fields = projected_arrow_fields.clone();
                        let filter = filter.clone();
                        let get_projected_field_at_output_index =
                            get_projected_field_at_output_index.clone();

                        async move {
                            // This is exact as we have already taken out the remainder.
                            (offset
                                ..offset
                                    .saturating_add(cols_per_thread)
                                    .min(projected_arrow_fields.len()))
                                .map(|i| {
                                    let projection = &projected_arrow_fields
                                        [get_projected_field_at_output_index(i)];

                                    let (col, pred_true_mask) = decode_column(
                                        projection.arrow_field(),
                                        &row_group_data,
                                        filter.clone(),
                                        expected_num_rows,
                                    )?;

                                    let col = projection.apply_transform(col)?;

                                    Ok((col, pred_true_mask))
                                })
                                .collect::<PolarsResult<UnitVec<_>>>()
                        }
                    }),
            )
        };

        for fut in task_handles {
            out_vec.extend(fut.await?.into_iter().map(|(c, _)| c));
        }

        Ok(())
    }
}

fn decode_column(
    arrow_field: &ArrowField,
    row_group_data: &RowGroupData,
    filter: Option<polars_parquet::read::Filter>,
    expected_num_rows: usize,
) -> PolarsResult<(Column, Bitmap)> {
    let Some(iter) = row_group_data
        .row_group_metadata
        .columns_under_root_iter(&arrow_field.name)
    else {
        return Ok((
            Column::full_null(
                arrow_field.name.clone(),
                expected_num_rows,
                &DataType::from_arrow_field(arrow_field),
            ),
            Bitmap::default(),
        ));
    };

    let columns_to_deserialize = iter
        .map(|col_md| {
            let byte_range = col_md.byte_range();

            (
                col_md,
                row_group_data
                    .fetched_bytes
                    .get_range(byte_range.start as usize..byte_range.end as usize),
            )
        })
        .collect::<Vec<_>>();

    let skip_num_rows_check = matches!(filter, Some(Filter::Predicate(_)));

    let (arrays, pred_true_mask) = polars_io::prelude::_internal::to_deserializer(
        columns_to_deserialize,
        arrow_field.clone(),
        filter,
    )?;

    if !skip_num_rows_check {
        let num_rows = arrays.iter().map(|array| array.len()).sum::<usize>();
        assert_eq!(num_rows, expected_num_rows);
    }

    let mut series = Series::try_from((arrow_field, arrays))?;

    if let Some(col_idxs) = row_group_data
        .row_group_metadata
        .columns_idxs_under_root_iter(&arrow_field.name)
    {
        if col_idxs.len() == 1 {
            try_set_sorted_flag(&mut series, col_idxs[0], &row_group_data.sorting_map);
        }
    }

    // TODO: Also load in the metadata.

    Ok((series.into_column(), pred_true_mask))
}

/// Filters columns, in parallel depending number of rows / columns.
async fn filter_cols(
    cols: Vec<Column>,
    mask: &BooleanChunked,
    target_values_per_thread: usize,
) -> PolarsResult<Vec<Column>> {
    if cols.is_empty() {
        return Ok(cols);
    }

    let cols_per_thread = calc_cols_per_thread(cols[0].len(), target_values_per_thread);
    let mut out_vec = Vec::with_capacity(cols.len());
    let cols = Arc::new(cols);
    let mask = mask.clone();

    let task_handles = {
        let cols = &cols;
        let mask = &mask;

        parallelize_first_to_local(
            TaskPriority::Low,
            (0..cols.len()).step_by(cols_per_thread).map(move |offset| {
                let cols = cols.clone();
                let mask = mask.clone();
                async move {
                    (offset..offset.saturating_add(cols_per_thread).min(cols.len()))
                        .map(|i| cols[i].filter(&mask))
                        .collect::<PolarsResult<UnitVec<_>>>()
                }
            }),
        )
    };

    for fut in task_handles {
        out_vec.extend(fut.await?)
    }

    Ok(out_vec)
}

fn calc_cols_per_thread(n_rows_per_col: usize, target_n_rows_per_thread: usize) -> usize {
    if n_rows_per_col == 0 {
        return usize::MAX;
    }

    let n = target_n_rows_per_thread / n_rows_per_col;
    let floor_distance = target_n_rows_per_thread % n_rows_per_col;
    let ceil_distance = n_rows_per_col - floor_distance;

    if floor_distance <= ceil_distance {
        n.max(1)
    } else {
        n + 1
    }
}

// Pre-filtered

fn decode_column_in_filter(
    arrow_field: &ArrowField,
    use_column_predicates: bool,
    column_predicates: &ColumnPredicates,
    row_group_data: &RowGroupData,
    projection_height: usize,
) -> PolarsResult<(Column, Bitmap)> {
    let mut filter = None;
    let mut constant = None;
    if use_column_predicates {
        if let Some((column_predicate, specialized)) =
            column_predicates.predicates.get(&arrow_field.name)
        {
            constant = specialized.as_ref().and_then(|s| match s {
                SpecializedColumnPredicate::Equal(sc) if !sc.is_null() => Some(sc),
                _ => None,
            });

            let p = ColumnPredicateExpr::new(
                arrow_field.name.clone(),
                DataType::from_arrow_field(arrow_field),
                column_predicate.clone(),
                specialized.clone(),
            );
            filter = Some(Filter::Predicate(PredicateFilter {
                predicate: Arc::new(p) as _,
                include_values: constant.is_none(),
            }));
        }
    }
    let (mut c, m) = decode_column(arrow_field, row_group_data, filter, projection_height)?;

    if let Some(constant) = constant {
        c = Column::new_scalar(c.name().clone(), constant.clone(), m.set_bits());
    }

    Ok((c, m))
}

struct EagerPredicateResult {
    predicate_idx: usize,
    column: Column,
    mask: Bitmap,
    elapsed_ns: f64,
}

struct MaskedPredicateResult {
    predicate_idx: usize,
    column: Column,
    mask: Bitmap,
    elapsed_ns: f64,
}

fn combine_masks(masks: impl IntoIterator<Item = Bitmap>) -> Bitmap {
    let mut masks = masks.into_iter();
    let Some(first) = masks.next() else {
        return Bitmap::new();
    };

    let mut combined = MutableBitmap::new();
    combined.extend_from_bitmap(&first);
    for mask in masks {
        <&mut MutableBitmap as std::ops::BitAndAssign<&Bitmap>>::bitand_assign(
            &mut &mut combined,
            &mask,
        );
    }
    combined.freeze()
}

fn boolean_mask_to_bitmap(mut mask: BooleanChunked) -> Bitmap {
    mask.rechunk_mut();
    let array = mask.downcast_as_array();
    match array.validity() {
        None => array.values().clone(),
        Some(validity) => array.values() & validity,
    }
}

/// Expands `selected` back into the coordinate space of `candidate`, placing its bits at the set
/// positions of `candidate` and writing false everywhere else.
fn deposit_mask(candidate: &Bitmap, selected: &Bitmap) -> Bitmap {
    assert_eq!(candidate.set_bits(), selected.len());

    let mut selected_iter = selected.iter();
    let mut builder = BitmapBuilder::with_capacity(candidate.len());
    let mut chunks = candidate.chunks::<u64>();

    for mut candidate_word in &mut chunks {
        let mut output_word = 0u64;
        while candidate_word != 0 {
            let bit = candidate_word.trailing_zeros() as usize;
            if selected_iter.next().unwrap() {
                output_word |= 1 << bit;
            }
            candidate_word &= candidate_word - 1;
        }
        // SAFETY: The builder was allocated for `candidate.len()` bits and each full chunk adds
        // exactly 64 bits.
        unsafe { builder.push_word_with_len_unchecked(output_word, 64) };
    }

    let remainder_len = chunks.remainder_len();
    if remainder_len > 0 {
        let mut candidate_word = chunks.remainder();
        let mut output_word = 0u64;
        while candidate_word != 0 {
            let bit = candidate_word.trailing_zeros() as usize;
            if selected_iter.next().unwrap() {
                output_word |= 1 << bit;
            }
            candidate_word &= candidate_word - 1;
        }
        // SAFETY: `remainder_len` is the exact unused capacity and is at most 63.
        unsafe { builder.push_word_with_len_unchecked(output_word, remainder_len) };
    }

    debug_assert!(selected_iter.next().is_none());
    builder.freeze()
}

fn filter_eager_predicate_columns(
    mut predicates: Vec<EagerPredicateResult>,
) -> PolarsResult<(Vec<Column>, Bitmap)> {
    predicates.sort_unstable_by_key(|p| p.predicate_idx);
    let final_mask = combine_masks(predicates.iter().map(|p| p.mask.clone()));
    let final_mask_ca = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, final_mask.clone());

    let columns = predicates
        .into_iter()
        .map(|p| {
            let predicate_mask = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, p.mask);
            let keep = final_mask_ca.filter(&predicate_mask)?;
            p.column.filter(&keep)
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    Ok((columns, final_mask))
}

fn masked_cost_factor(
    predicate_idx: usize,
    predicate_field_indices: &[usize],
    projected_arrow_fields: &[ArrowFieldProjection],
    scan_predicate: &ScanIOPredicate,
    row_group_data: &RowGroupData,
) -> f64 {
    let projection = &projected_arrow_fields[predicate_field_indices[predicate_idx]];
    let field = projection.arrow_field();
    let dtype = DataType::from_arrow_field(field);
    let specialized = scan_predicate
        .column_predicates
        .predicates
        .get(field.name.as_str())
        .and_then(|(_, specialized)| specialized.as_ref());
    let dictionary_backed = row_group_data
        .row_group_metadata
        .columns_under_root_iter(&field.name)
        .is_some_and(|mut columns| columns.any(|column| column.dictionary_page_offset().is_some()));

    if (dtype.is_string() || dtype.is_binary())
        && dictionary_backed
        && matches!(
            specialized,
            Some(SpecializedColumnPredicate::Equal(_) | SpecializedColumnPredicate::EqualOneOf(_))
        )
    {
        4.0
    } else if dtype.is_string() || dtype.is_binary() {
        2.0
    } else {
        1.25
    }
}

impl RowGroupDecoder {
    async fn decode_eager_predicates(
        &self,
        predicate_indices: Arc<[usize]>,
        scan_predicate: &ScanIOPredicate,
        row_group_data: &Arc<RowGroupData>,
        projection_height: usize,
    ) -> PolarsResult<Vec<EagerPredicateResult>> {
        let num_predicates = predicate_indices.len();
        let measure_elapsed = self.adaptive_predicate_state.is_some();
        let cols_per_thread = (predicate_indices.len().div_ceil(self.num_pipelines)).max(1);
        let task_handles = {
            let predicate_field_indices = self.predicate_field_indices.clone();
            let projected_arrow_fields = self.projected_arrow_fields.clone();
            let row_group_data = row_group_data.clone();
            let column_predicates = scan_predicate.column_predicates.clone();

            parallelize_first_to_local(
                TaskPriority::Low,
                (0..predicate_indices.len())
                    .step_by(cols_per_thread)
                    .map(move |offset| {
                        let predicate_indices = predicate_indices.clone();
                        let predicate_field_indices = predicate_field_indices.clone();
                        let projected_arrow_fields = projected_arrow_fields.clone();
                        let row_group_data = row_group_data.clone();
                        let column_predicates = column_predicates.clone();

                        async move {
                            (offset
                                ..offset
                                    .saturating_add(cols_per_thread)
                                    .min(predicate_indices.len()))
                                .map(|i| {
                                    let predicate_idx = predicate_indices[i];
                                    let projection = &projected_arrow_fields
                                        [predicate_field_indices[predicate_idx]];
                                    let start = measure_elapsed.then(Instant::now);
                                    let (column, mask) = decode_column_in_filter(
                                        projection.arrow_field(),
                                        true,
                                        column_predicates.as_ref(),
                                        row_group_data.as_ref(),
                                        projection_height,
                                    )?;
                                    let column = projection.apply_transform(column)?;

                                    Ok(EagerPredicateResult {
                                        predicate_idx,
                                        column,
                                        mask,
                                        elapsed_ns: start
                                            .map(|start| start.elapsed().as_nanos() as f64)
                                            .unwrap_or(0.0),
                                    })
                                })
                                .collect::<PolarsResult<UnitVec<_>>>()
                        }
                    }),
            )
        };

        let mut out = Vec::with_capacity(num_predicates);
        for task in task_handles {
            out.extend(task.await?);
        }
        Ok(out)
    }

    async fn decode_masked_predicates(
        &self,
        predicate_indices: Arc<[usize]>,
        scan_predicate: &ScanIOPredicate,
        row_group_data: &Arc<RowGroupData>,
        candidate_mask: &Bitmap,
    ) -> PolarsResult<Vec<MaskedPredicateResult>> {
        let num_predicates = predicate_indices.len();
        let expected_num_rows = candidate_mask.set_bits();
        let cols_per_thread = (predicate_indices.len().div_ceil(self.num_pipelines)).max(1);
        let task_handles = {
            let predicate_field_indices = self.predicate_field_indices.clone();
            let projected_arrow_fields = self.projected_arrow_fields.clone();
            let row_group_data = row_group_data.clone();
            let column_predicates = scan_predicate.column_predicates.clone();
            let candidate_mask = candidate_mask.clone();

            parallelize_first_to_local(
                TaskPriority::Low,
                (0..predicate_indices.len())
                    .step_by(cols_per_thread)
                    .map(move |offset| {
                        let predicate_indices = predicate_indices.clone();
                        let predicate_field_indices = predicate_field_indices.clone();
                        let projected_arrow_fields = projected_arrow_fields.clone();
                        let row_group_data = row_group_data.clone();
                        let column_predicates = column_predicates.clone();
                        let candidate_mask = candidate_mask.clone();

                        async move {
                            (offset
                                ..offset
                                    .saturating_add(cols_per_thread)
                                    .min(predicate_indices.len()))
                                .map(|i| {
                                    let predicate_idx = predicate_indices[i];
                                    let projection = &projected_arrow_fields
                                        [predicate_field_indices[predicate_idx]];
                                    let start = Instant::now();
                                    let (column, _) = decode_column(
                                        projection.arrow_field(),
                                        row_group_data.as_ref(),
                                        Some(Filter::Mask(candidate_mask.clone())),
                                        expected_num_rows,
                                    )?;
                                    let column = projection.apply_transform(column)?;
                                    let (predicate, _) = column_predicates
                                        .predicates
                                        .get(projection.output_name())
                                        .unwrap();
                                    let df = unsafe {
                                        DataFrame::new_unchecked(
                                            expected_num_rows,
                                            vec![column.clone()],
                                        )
                                    };
                                    let mask = predicate.evaluate_io(&df)?;
                                    let mask = boolean_mask_to_bitmap(mask.bool()?.clone());
                                    let mask_ca = BooleanChunked::from_bitmap(
                                        PlSmallStr::EMPTY,
                                        mask.clone(),
                                    );
                                    let column = column.filter(&mask_ca)?;

                                    Ok(MaskedPredicateResult {
                                        predicate_idx,
                                        column,
                                        mask,
                                        elapsed_ns: start.elapsed().as_nanos() as f64,
                                    })
                                })
                                .collect::<PolarsResult<UnitVec<_>>>()
                        }
                    }),
            )
        };

        let mut out = Vec::with_capacity(num_predicates);
        for task in task_handles {
            out.extend(task.await?);
        }
        Ok(out)
    }

    async fn decode_adaptive_predicates(
        &self,
        seed_indices: Vec<usize>,
        scan_predicate: &ScanIOPredicate,
        row_group_data: &Arc<RowGroupData>,
        projection_height: usize,
    ) -> PolarsResult<(Vec<Column>, Bitmap)> {
        let state = self.adaptive_predicate_state.as_ref().unwrap();
        let seed_indices: Arc<[usize]> = seed_indices.into();
        let mut eager = self
            .decode_eager_predicates(
                seed_indices.clone(),
                scan_predicate,
                row_group_data,
                projection_height,
            )
            .await?;

        for result in &eager {
            state.record_eager(
                result.predicate_idx,
                result.elapsed_ns,
                projection_height,
                result.mask.set_bits(),
            );
        }

        let seed_mask = combine_masks(eager.iter().map(|p| p.mask.clone()));
        let seed_rows = seed_mask.set_bits();
        let seed_density = if projection_height == 0 {
            0.0
        } else {
            seed_rows as f64 / projection_height as f64
        };
        let remaining = (0..self.predicate_field_indices.len())
            .filter(|idx| !seed_indices.contains(idx))
            .collect::<Vec<_>>();

        if state.verbose {
            eprintln!(
                "[ParquetFileReader]: Adaptive predicate seed {:?}, density: {:.4}",
                seed_indices, seed_density
            );
        }

        if seed_density > ADAPTIVE_MAX_ACTUAL_SEED_DENSITY {
            if state.verbose {
                eprintln!("[ParquetFileReader]: Adaptive predicate fallback to eager remainder");
            }
            let remainder = self
                .decode_eager_predicates(
                    remaining.into(),
                    scan_predicate,
                    row_group_data,
                    projection_height,
                )
                .await?;
            for result in &remainder {
                state.record_eager(
                    result.predicate_idx,
                    result.elapsed_ns,
                    projection_height,
                    result.mask.set_bits(),
                );
            }
            eager.extend(remainder);
            return filter_eager_predicate_columns(eager);
        }

        let mut masked = self
            .decode_masked_predicates(remaining.into(), scan_predicate, row_group_data, &seed_mask)
            .await?;
        for result in &masked {
            state.record_masked(
                result.predicate_idx,
                result.elapsed_ns,
                seed_rows,
                result.mask.set_bits(),
            );
        }

        let combined_local_mask = combine_masks(masked.iter().map(|p| p.mask.clone()));
        let final_mask = deposit_mask(&seed_mask, &combined_local_mask);
        let final_mask_ca = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, final_mask.clone());
        let combined_local_mask_ca =
            BooleanChunked::from_bitmap(PlSmallStr::EMPTY, combined_local_mask);
        let mut columns = (0..self.predicate_field_indices.len())
            .map(|_| None)
            .collect::<Vec<Option<Column>>>();

        for result in eager.drain(..) {
            let predicate_mask = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, result.mask);
            let keep = final_mask_ca.filter(&predicate_mask)?;
            columns[result.predicate_idx] = Some(result.column.filter(&keep)?);
        }
        for result in masked.drain(..) {
            let predicate_mask = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, result.mask);
            let keep = combined_local_mask_ca.filter(&predicate_mask)?;
            columns[result.predicate_idx] = Some(result.column.filter(&keep)?);
        }

        if state.verbose {
            eprintln!(
                "[ParquetFileReader]: Adaptive predicate masked {} / {} rows",
                seed_rows, projection_height
            );
        }

        Ok((
            columns.into_iter().map(Option::unwrap).collect(),
            final_mask,
        ))
    }
}

impl RowGroupDecoder {
    async fn row_group_data_to_df_prefiltered(
        &self,
        row_group_data: RowGroupData,
    ) -> PolarsResult<DataFrame> {
        debug_assert!(row_group_data.slice.is_none()); // Invariant of the optimizer.
        assert!(self.predicate_field_indices.len() <= self.projected_arrow_fields.len());

        let row_group_data = Arc::new(row_group_data);
        let projection_height = row_group_data.row_group_metadata.num_rows();

        let mut live_columns = Vec::with_capacity(
            self.row_index.is_some() as usize
                + self.predicate_field_indices.len()
                + self.non_predicate_field_indices.len(),
        );
        if let Some(s) = self.materialize_row_index(
            row_group_data.as_ref(),
            0..row_group_data.row_group_metadata.num_rows(),
        )? {
            live_columns.push(s);
        }

        let scan_predicate = self.predicate.as_ref().unwrap();

        let use_column_predicates = self.allow_column_predicates
            && !row_group_data
                .row_group_metadata
                .parquet_columns()
                .iter()
                .any(|c| {
                    matches!(
                        c.descriptor().descriptor.primitive_type.logical_type,
                        Some(PrimitiveLogicalType::Float16)
                    )
                });

        let (live_df_filtered, mut mask) = if use_column_predicates {
            assert!(scan_predicate.column_predicates.is_sumwise_complete);
            let adaptive_seed = self.adaptive_predicate_state.as_ref().and_then(|state| {
                let masked_cost_factors = (0..self.predicate_field_indices.len())
                    .map(|predicate_idx| {
                        masked_cost_factor(
                            predicate_idx,
                            &self.predicate_field_indices,
                            &self.projected_arrow_fields,
                            scan_predicate,
                            row_group_data.as_ref(),
                        )
                    })
                    .collect::<Vec<_>>();
                state.choose_seed(&masked_cost_factors)
            });

            let (columns, mask) = if let Some(seed) = adaptive_seed {
                self.decode_adaptive_predicates(
                    seed,
                    scan_predicate,
                    &row_group_data,
                    projection_height,
                )
                .await?
            } else {
                let all_predicates: Arc<[usize]> = (0..self.predicate_field_indices.len())
                    .collect::<Vec<_>>()
                    .into();
                let eager = self
                    .decode_eager_predicates(
                        all_predicates,
                        scan_predicate,
                        &row_group_data,
                        projection_height,
                    )
                    .await?;
                if let Some(state) = self.adaptive_predicate_state.as_ref() {
                    for result in &eager {
                        state.record_eager(
                            result.predicate_idx,
                            result.elapsed_ns,
                            projection_height,
                            result.mask.set_bits(),
                        );
                    }
                }
                filter_eager_predicate_columns(eager)?
            };
            let height = mask.set_bits();
            (
                unsafe { DataFrame::new_unchecked(height, columns) },
                BooleanChunked::from_bitmap(PlSmallStr::EMPTY, mask),
            )
        } else {
            let cols_per_thread = (self
                .predicate_field_indices
                .len()
                .div_ceil(self.num_pipelines))
            .max(1);
            let task_handles = {
                let predicate_field_indices = self.predicate_field_indices.clone();
                let projected_arrow_fields = self.projected_arrow_fields.clone();
                let row_group_data = row_group_data.clone();
                let column_predicates = scan_predicate.column_predicates.clone();

                parallelize_first_to_local(
                    TaskPriority::Low,
                    (0..self.predicate_field_indices.len())
                        .step_by(cols_per_thread)
                        .map(move |offset| {
                            let row_group_data = row_group_data.clone();
                            let predicate_field_indices = predicate_field_indices.clone();
                            let projected_arrow_fields = projected_arrow_fields.clone();
                            let column_predicates = column_predicates.clone();

                            async move {
                                (offset
                                    ..offset
                                        .saturating_add(cols_per_thread)
                                        .min(predicate_field_indices.len()))
                                    .map(|i| {
                                        let projection =
                                            &projected_arrow_fields[predicate_field_indices[i]];
                                        let (column, _) = decode_column_in_filter(
                                            projection.arrow_field(),
                                            false,
                                            column_predicates.as_ref(),
                                            row_group_data.as_ref(),
                                            projection_height,
                                        )?;
                                        projection.apply_transform(column)
                                    })
                                    .collect::<PolarsResult<UnitVec<_>>>()
                            }
                        }),
                )
            };
            for task in task_handles {
                live_columns.extend(task.await?);
            }

            let mut live_df = unsafe {
                DataFrame::new_unchecked(row_group_data.row_group_metadata.num_rows(), live_columns)
            };

            let mask = scan_predicate.predicate.evaluate_io(&live_df)?;
            let mask = mask.bool().unwrap();

            unsafe {
                live_df.columns_mut().truncate(
                    self.row_index.is_some() as usize + self.predicate_field_indices.len(),
                )
            }

            let filtered =
                filter_cols(live_df.into_columns(), mask, self.target_values_per_thread).await?;

            let filtered_height = if let Some(fst) = filtered.first() {
                fst.len()
            } else {
                mask.num_trues()
            };

            (
                unsafe { DataFrame::new_unchecked(filtered_height, filtered) },
                mask.clone(),
            )
        };

        if self.non_predicate_field_indices.is_empty() {
            // User or test may have explicitly requested prefiltering
            return Ok(live_df_filtered);
        }

        mask.rechunk_mut();
        let mask_bitmap = mask.downcast_as_array();
        let mask_bitmap = match mask_bitmap.validity() {
            None => mask_bitmap.values().clone(),
            Some(v) => mask_bitmap.values() & v,
        };

        assert_eq!(mask_bitmap.len(), projection_height);

        let expected_num_rows = mask_bitmap.set_bits();

        let cols_per_thread = (self
            .predicate_field_indices
            .len()
            .div_ceil(self.num_pipelines))
        .max(1);

        let task_handles = {
            let non_predicate_field_indices = self.non_predicate_field_indices.clone();
            let non_predicate_len = non_predicate_field_indices.len();
            let projected_arrow_fields = self.projected_arrow_fields.clone();
            let row_group_data = row_group_data.clone();

            parallelize_first_to_local(
                TaskPriority::Low,
                (0..non_predicate_len)
                    .step_by(cols_per_thread)
                    .map(move |offset| {
                        let row_group_data = row_group_data.clone();
                        let non_predicate_field_indices = non_predicate_field_indices.clone();
                        let projected_arrow_fields = projected_arrow_fields.clone();
                        let mask = mask.clone();
                        let mask_bitmap = mask_bitmap.clone();

                        async move {
                            (offset
                                ..offset
                                    .saturating_add(cols_per_thread)
                                    .min(non_predicate_len))
                                .map(|i| {
                                    let projection =
                                        &projected_arrow_fields[non_predicate_field_indices[i]];

                                    let col = decode_column_prefiltered(
                                        projection.arrow_field(),
                                        row_group_data.as_ref(),
                                        &mask,
                                        &mask_bitmap,
                                        expected_num_rows,
                                    )?;

                                    projection.apply_transform(col)
                                })
                                .collect::<PolarsResult<UnitVec<_>>>()
                        }
                    }),
            )
        };

        drop(row_group_data);

        let live_columns = live_df_filtered.into_columns();

        let mut dead_cols = Vec::with_capacity(self.non_predicate_field_indices.len());
        for fut in task_handles {
            dead_cols.extend(fut.await?);
        }

        let mut merged = live_columns;
        merged.extend(dead_cols);
        let df = unsafe { DataFrame::new_unchecked(expected_num_rows, merged) };
        Ok(df)
    }
}

fn decode_column_prefiltered(
    arrow_field: &ArrowField,
    row_group_data: &RowGroupData,
    mask: &BooleanChunked,
    mask_bitmap: &Bitmap,
    expected_num_rows: usize,
) -> PolarsResult<Column> {
    let Some(iter) = row_group_data
        .row_group_metadata
        .columns_under_root_iter(&arrow_field.name)
    else {
        return Ok(Column::full_null(
            arrow_field.name.clone(),
            expected_num_rows,
            &DataType::from_arrow_field(arrow_field),
        ));
    };

    let columns_to_deserialize = iter
        .map(|col_md| {
            let byte_range = col_md.byte_range();

            (
                col_md,
                row_group_data
                    .fetched_bytes
                    .get_range(byte_range.start as usize..byte_range.end as usize),
            )
        })
        .collect::<Vec<_>>();

    let prefilter = !arrow_field.dtype.is_nested();

    let deserialize_filter =
        prefilter.then(|| polars_parquet::read::Filter::Mask(mask_bitmap.clone()));

    let (array, _) = polars_io::prelude::_internal::to_deserializer(
        columns_to_deserialize,
        arrow_field.clone(),
        deserialize_filter,
    )?;

    let mut series = Series::try_from((arrow_field, array))?;

    if let Some(col_idxs) = row_group_data
        .row_group_metadata
        .columns_idxs_under_root_iter(&arrow_field.name)
    {
        if col_idxs.len() == 1 {
            try_set_sorted_flag(&mut series, col_idxs[0], &row_group_data.sorting_map);
        }
    }

    let series = if !prefilter {
        series.filter(mask)?
    } else {
        series
    };

    assert_eq!(series.len(), expected_num_rows);

    Ok(series.into_column())
}

#[cfg(test)]
mod tests {
    use polars_core::utils::arrow::bitmap::Bitmap;

    use super::{AdaptivePredicateState, deposit_mask};

    #[test]
    fn test_calc_cols_per_thread() {
        use super::calc_cols_per_thread;

        assert_eq!(
            [
                calc_cols_per_thread(0, 5),
                calc_cols_per_thread(1, 5),
                calc_cols_per_thread(2, 5),
                calc_cols_per_thread(3, 5),
                calc_cols_per_thread(4, 5),
                calc_cols_per_thread(5, 5),
            ],
            [usize::MAX, 5, 2, 2, 1, 1]
        );

        assert_eq!(
            [
                calc_cols_per_thread(11_184_810, 16_777_216),
                calc_cols_per_thread(11_184_811, 16_777_216),
            ],
            [2, 1]
        );

        assert_eq!(
            [
                calc_cols_per_thread(0, 0),
                calc_cols_per_thread(0, 99),
                calc_cols_per_thread(99, 0),
                calc_cols_per_thread(99, 99),
            ],
            [usize::MAX, usize::MAX, 1, 1],
        )
    }

    #[test]
    fn test_deposit_mask() {
        let candidate = Bitmap::from([
            false, true, true, false, false, true, false, true, true, false, true,
        ]);
        let selected = Bitmap::from([true, false, true, true, false, true]);
        let expected = [
            false, true, false, false, false, true, false, true, false, false, true,
        ];

        assert_eq!(
            deposit_mask(&candidate, &selected)
                .iter()
                .collect::<Vec<_>>(),
            expected
        );

        let candidate = Bitmap::from([true; 130]);
        let selected = Bitmap::from_iter((0..130).map(|i| i % 3 == 0));
        assert_eq!(deposit_mask(&candidate, &selected), selected);

        let candidate = Bitmap::from([false; 67]);
        assert_eq!(deposit_mask(&candidate, &Bitmap::new()), candidate);

        let mut candidate = Bitmap::from_iter((0..140).map(|i| i % 5 == 1 || i % 7 == 0));
        candidate.slice(5, 129);
        let selected = Bitmap::from_iter((0..candidate.set_bits()).map(|i| i % 2 == 0));
        let mut selected_iter = selected.iter();
        let expected = Bitmap::from_iter(
            candidate
                .iter()
                .map(|is_candidate| is_candidate && selected_iter.next().unwrap()),
        );
        assert_eq!(deposit_mask(&candidate, &selected), expected);
    }

    #[test]
    fn test_adaptive_predicate_selects_selective_seed() {
        let state = AdaptivePredicateState::new(3, false);
        for _ in 0..4 {
            state.record_eager(0, 10_000.0, 1_000, 200);
            state.record_eager(1, 12_000.0, 1_000, 200);
            state.record_eager(2, 100_000.0, 1_000, 600);
        }

        assert_eq!(state.choose_seed(&[1.25, 1.25, 1.25]), Some(vec![0, 1]));
    }

    #[test]
    fn test_adaptive_predicate_waits_for_training_and_rejects_weak_seed() {
        let state = AdaptivePredicateState::new(3, false);
        for _ in 0..3 {
            state.record_eager(0, 10_000.0, 1_000, 900);
            state.record_eager(1, 10_000.0, 1_000, 900);
            state.record_eager(2, 10_000.0, 1_000, 900);
        }
        assert_eq!(state.choose_seed(&[1.25; 3]), None);

        state.record_eager(0, 10_000.0, 1_000, 900);
        state.record_eager(1, 10_000.0, 1_000, 900);
        state.record_eager(2, 10_000.0, 1_000, 900);
        assert_eq!(state.choose_seed(&[1.25; 3]), None);
    }
}
