//! Negation normal form for boolean `AExpr` trees.

use polars_utils::aliases::{InitHashMaps, PlHashMap};
use polars_utils::arena::{Arena, Node};
use recursive::recursive;

use super::{AExpr, AExprBuilder, IRBooleanFunction, IRFunctionExpr};
use crate::prelude::Operator;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct State {
    node: Node,
    negated: bool,
}

/// Rewrite the boolean expression root into negation normal form.
///
/// In the returned expression, Not only occurs immediately above atoms.
/// Note that both `LogicalAnd` and `LogicalOr` are treat as atoms by this
/// implementation since without type information it is unsafe to push
/// `Not` through them: we do not know if their inputs are also boolean.
pub(crate) fn to_nnf(root: Node, expr_arena: &mut Arena<AExpr>) -> Node {
    let mut cache = PlHashMap::new();
    to_nnf_impl(
        State {
            node: root,
            negated: false,
        },
        expr_arena,
        &mut cache,
    )
}

#[recursive]
fn to_nnf_impl(
    state: State,
    expr_arena: &mut Arena<AExpr>,
    cache: &mut PlHashMap<State, Node>,
) -> Node {
    if let Some(&node) = cache.get(&state) {
        return node;
    }

    let node = match expr_arena.get(state.node) {
        AExpr::Function {
            input,
            function: IRFunctionExpr::Boolean(IRBooleanFunction::Not),
            ..
        } => {
            debug_assert!(
                input.len() == 1,
                "Malformed IRBooleanFunction::Not expression"
            );
            let input = input[0].node();
            if state.negated || !is_atom(input, expr_arena) {
                to_nnf_impl(
                    State {
                        node: input,
                        negated: !state.negated,
                    },
                    expr_arena,
                    cache,
                )
            } else {
                state.node
            }
        },
        AExpr::BinaryExpr { left, op, right } if matches!(op, Operator::And | Operator::Or) => {
            let (old_left, op, old_right) = (*left, *op, *right);
            let left = to_nnf_impl(
                State {
                    node: old_left,
                    negated: state.negated,
                },
                expr_arena,
                cache,
            );
            let right = to_nnf_impl(
                State {
                    node: old_right,
                    negated: state.negated,
                },
                expr_arena,
                cache,
            );
            if !state.negated && left == old_left && right == old_right {
                state.node
            } else {
                let op = if state.negated {
                    match op {
                        Operator::And => Operator::Or,
                        Operator::Or => Operator::And,
                        _ => unreachable!(),
                    }
                } else {
                    op
                };
                expr_arena.add(AExpr::BinaryExpr { left, op, right })
            }
        },
        _ => {
            if state.negated {
                let input = AExprBuilder::new_from_node(state.node).expr_ir_retain_name(expr_arena);
                AExprBuilder::function(
                    vec![input],
                    IRFunctionExpr::Boolean(IRBooleanFunction::Not),
                    expr_arena,
                )
                .node()
            } else {
                state.node
            }
        },
    };
    cache.insert(state, node);
    node
}

fn is_atom(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    match expr_arena.get(node) {
        AExpr::BinaryExpr { op, .. } => !matches!(op, Operator::And | Operator::Or),
        AExpr::Function {
            input,
            function: IRFunctionExpr::Boolean(IRBooleanFunction::Not),
            ..
        } => {
            debug_assert!(input.len() == 1, "Malformed Not expression");
            false
        },
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plans::ExprIR;
    use crate::plans::aexpr::AExprBuilder;

    fn assert_binary(node: Node, expected_op: Operator, expr_arena: &Arena<AExpr>) -> (Node, Node) {
        let AExpr::BinaryExpr { left, op, right } = expr_arena.get(node) else {
            panic!("expected binary expression, got {:?}", expr_arena.get(node));
        };
        assert_eq!(*op, expected_op);
        (*left, *right)
    }

    fn assert_not(node: Node, expected_input: Node, expr_arena: &Arena<AExpr>) {
        let AExpr::Function {
            input,
            function: IRFunctionExpr::Boolean(IRBooleanFunction::Not),
            ..
        } = expr_arena.get(node)
        else {
            panic!("expected Not expression, got {:?}", expr_arena.get(node));
        };
        assert_eq!(
            input.as_slice(),
            &[ExprIR::from_node(expected_input, expr_arena)]
        );
    }

    #[test]
    fn applies_de_morgan_and_eliminates_double_negation() {
        let mut expr_arena = Arena::new();
        let a = AExprBuilder::col("a", &mut expr_arena).node();
        let b = AExprBuilder::col("b", &mut expr_arena).node();
        let c = AExprBuilder::col("c", &mut expr_arena).node();

        // NOT(a AND (b OR NOT(c)))
        let not_c = AExprBuilder::new_from_node(c).not(&mut expr_arena);
        let inner = AExprBuilder::new_from_node(b).or(not_c, &mut expr_arena);
        let root = AExprBuilder::new_from_node(a)
            .and(inner, &mut expr_arena)
            .not(&mut expr_arena)
            .node();

        let nnf = to_nnf(root, &mut expr_arena);

        // NOT(a) OR (NOT(b) AND c)
        let (not_a, rhs) = assert_binary(nnf, Operator::Or, &expr_arena);
        assert_not(not_a, a, &expr_arena);
        let (not_b, out_c) = assert_binary(rhs, Operator::And, &expr_arena);
        assert_not(not_b, b, &expr_arena);
        assert_eq!(out_c, c);

        // Running the utility again is allocation-free and preserves the root.
        let arena_len = expr_arena.len();
        assert_eq!(to_nnf(nnf, &mut expr_arena), nnf);
        assert_eq!(expr_arena.len(), arena_len);
    }

    #[test]
    fn maps_each_boolean_connective_to_its_dual() {
        let cases = [(Operator::And, Operator::Or), (Operator::Or, Operator::And)];

        for (input_op, expected_op) in cases {
            let mut expr_arena = Arena::new();
            let left = AExprBuilder::col("left", &mut expr_arena).node();
            let right = AExprBuilder::col("right", &mut expr_arena).node();
            let binary = expr_arena.add(AExpr::BinaryExpr {
                left,
                op: input_op,
                right,
            });
            let root = AExprBuilder::new_from_node(binary)
                .not(&mut expr_arena)
                .node();

            let nnf = to_nnf(root, &mut expr_arena);
            let (not_left, not_right) = assert_binary(nnf, expected_op, &expr_arena);
            assert_not(not_left, left, &expr_arena);
            assert_not(not_right, right, &expr_arena);
        }
    }

    #[test]
    fn treats_non_boolean_connectives_and_logical_casts_as_atoms() {
        let mut expr_arena = Arena::new();
        let left = AExprBuilder::col("left", &mut expr_arena).node();
        let right = AExprBuilder::col("right", &mut expr_arena).node();
        let xor = expr_arena.add(AExpr::BinaryExpr {
            left,
            op: Operator::Xor,
            right,
        });
        let root = AExprBuilder::new_from_node(xor).not(&mut expr_arena).node();
        let arena_len = expr_arena.len();

        assert_eq!(to_nnf(root, &mut expr_arena), root);
        assert_eq!(expr_arena.len(), arena_len);

        let logical_and = expr_arena.add(AExpr::BinaryExpr {
            left,
            op: Operator::LogicalAnd,
            right,
        });
        let root = AExprBuilder::new_from_node(logical_and)
            .not(&mut expr_arena)
            .node();
        let arena_len = expr_arena.len();

        assert_eq!(to_nnf(root, &mut expr_arena), root);
        assert_eq!(expr_arena.len(), arena_len);
    }
}
