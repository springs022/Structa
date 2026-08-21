import unittest
import random
from itertools import permutations

import cshogi as cs

from fixed_proof import (
    _minimum_rect_assignment,
    format_auto_fixed_pieces,
    prove_auto_fixed_sqs,
    sqs_to_rfs,
)
from validation import is_move_touching_fixed_sqs


class AssignmentTests(unittest.TestCase):
    def test_rectangular_assignment_uses_distinct_columns(self):
        self.assertEqual(
            3,
            _minimum_rect_assignment([
                [1, 8, 9],
                [1, 2, 9],
            ]),
        )

    def test_square_conversion_for_display(self):
        board = cs.Board()
        pawn_sq = next(
            sq for sq in range(81) if board.piece(sq) == cs.BPAWN
            and 77 in sqs_to_rfs({sq})
        )
        self.assertEqual({77}, sqs_to_rfs({pawn_sq}))

    def test_auto_fixed_display_includes_side_square_and_piece(self):
        board = cs.Board()
        squares = {
            sq for sq in range(81)
            if sqs_to_rfs({sq}) in ({51}, {59}, {69})
        }
        self.assertEqual(
            "後手51玉、先手59玉、先手69金",
            format_auto_fixed_pieces(board, squares),
        )

    def test_hungarian_matches_exhaustive_small_matrices(self):
        rng = random.Random(19460720)
        for rows in range(1, 5):
            cols = rows + 2
            for _ in range(20):
                costs = [
                    [rng.randrange(10) for _ in range(cols)]
                    for _ in range(rows)
                ]
                expected = min(
                    sum(costs[i][chosen[i]] for i in range(rows))
                    for chosen in permutations(range(cols), rows)
                )
                self.assertEqual(expected, _minimum_rect_assignment(costs))


class AutoFixedProofTests(unittest.TestCase):
    def test_one_move_problem_fixes_untouched_pieces(self):
        start = cs.Board()
        target = start.copy()
        target.push_usi("7g7f")

        fixed = prove_auto_fixed_sqs(start, target, 1)
        fixed_rfs = sqs_to_rfs(fixed)

        self.assertIn(59, fixed_rfs)   # 先手玉
        self.assertIn(51, fixed_rfs)   # 後手玉
        self.assertNotIn(77, fixed_rfs)  # 実際に動く歩

    def test_proven_fixed_squares_do_not_touch_known_solution(self):
        moves = ["7g7f", "3c3d", "2g2f", "8b3b"]
        start = cs.Board()
        target = start.copy()
        for usi in moves:
            target.push_usi(usi)

        fixed = prove_auto_fixed_sqs(start, target, len(moves))
        board = start.copy()
        for usi in moves:
            mv = board.move_from_usi(usi)
            self.assertFalse(
                is_move_touching_fixed_sqs(mv, fixed),
                f"既知解の {usi} が自動不動駒に触れています",
            )
            board.push(mv)

    def test_capture_promotion_and_drop_witness_is_preserved(self):
        moves = ["7g7f", "3c3d", "8h2b+", "3a2b", "B*4e"]
        start = cs.Board()
        target = start.copy()
        for usi in moves:
            mv = target.move_from_usi(usi)
            self.assertTrue(target.is_legal(mv), usi)
            target.push(mv)

        fixed = prove_auto_fixed_sqs(start, target, len(moves))
        witness = start.copy()
        for usi in moves:
            mv = witness.move_from_usi(usi)
            self.assertFalse(is_move_touching_fixed_sqs(mv, fixed), usi)
            witness.push(mv)

    def test_available_round_trip_prevents_false_proof(self):
        # 先手飛は9九-9八-9九、双方の玉も往復する余裕がある。
        start = cs.Board("4k4/9/9/9/9/9/9/9/R3K4 b - 1")
        target = start.copy()

        fixed_rfs = sqs_to_rfs(prove_auto_fixed_sqs(start, target, 4))

        self.assertNotIn(99, fixed_rfs)
        self.assertNotIn(59, fixed_rfs)
        self.assertNotIn(51, fixed_rfs)

    def test_manual_fixed_square_is_not_reported_as_auto(self):
        start = cs.Board()
        target = start.copy()
        target.push_usi("7g7f")
        king_sq = next(sq for sq in range(81) if start.piece(sq) == cs.BKING)

        fixed = prove_auto_fixed_sqs(start, target, 1, {king_sq})

        self.assertNotIn(king_sq, fixed)

    def test_random_legal_witnesses_never_touch_proven_squares(self):
        rng = random.Random(20260821)
        for sample in range(30):
            start = cs.Board()
            board = start.copy()
            moves = []
            length = 1 + sample % 8
            for _ in range(length):
                legal = list(board.legal_moves)
                if not legal:
                    break
                mv = rng.choice(legal)
                moves.append(cs.move_to_usi(mv))
                board.push(mv)
            target = board
            fixed = prove_auto_fixed_sqs(start, target, len(moves))

            witness = start.copy()
            for usi in moves:
                mv = witness.move_from_usi(usi)
                self.assertFalse(
                    is_move_touching_fixed_sqs(mv, fixed),
                    f"sample={sample}, move={usi}, fixed={sqs_to_rfs(fixed)}",
                )
                witness.push(mv)


if __name__ == "__main__":
    unittest.main()
