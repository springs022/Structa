import contextlib
import io
import unittest
from unittest.mock import patch

import cshogi as cs

from cost_calc import available_moves_for_side, corrected_need_moves_count
from search import find_all_paths_to_target


class SearchRegressionTests(unittest.TestCase):
    def test_one_move_solution_is_found(self):
        start = cs.Board()
        target = start.copy()
        target.push_usi("7g7f")

        with contextlib.redirect_stdout(io.StringIO()):
            solutions, stats, _, interrupted = find_all_paths_to_target(
                start,
                target,
                max_depth=1,
                limit=1,
                fixed_rfs=set(),
                tt_memory_mb=1,
                margin=0,
                first_move_index=0,
                previous_solutions=[],
                debug_usis=[],
            )

        self.assertFalse(interrupted)
        self.assertEqual(1, len(solutions))
        self.assertEqual(["7g7f"], [cs.move_to_usi(mv) for mv in solutions[0]])
        self.assertGreater(stats["total_nodes"], 0)

    def test_fixed_piece_check_is_skipped_when_not_configured(self):
        start = cs.Board()
        target = start.copy()
        target.push_usi("7g7f")

        with patch(
            "search.is_move_touching_fixed_piece",
            side_effect=AssertionError("空集合では呼ばない"),
        ), contextlib.redirect_stdout(io.StringIO()):
            solutions, _, _, _ = find_all_paths_to_target(
                start, target, 1, 1, set(), 1, 0, 0, [], []
            )

        self.assertEqual(1, len(solutions))

    def test_configured_fixed_piece_still_blocks_its_move(self):
        start = cs.Board()
        target = start.copy()
        target.push_usi("7g7f")

        with contextlib.redirect_stdout(io.StringIO()):
            solutions, _, _, _ = find_all_paths_to_target(
                start, target, 1, 1, {77}, 1, 0, 0, [], []
            )

        self.assertEqual([], solutions)

    def test_known_solution_prefixes_are_not_pruned(self):
        start = cs.Board()
        moves = ["7g7f", "3c3d", "8h3c+", "2a3c", "B*4e", "8b7b", "4e3d+"]
        target = start.copy()
        for usi in moves:
            target.push_usi(usi)

        board = start.copy()
        for depth, usi in enumerate(moves):
            board.push_usi(usi)
            remaining = len(moves) - depth - 1
            avail_s = available_moves_for_side(remaining, board.turn, cs.BLACK)
            avail_g = available_moves_for_side(remaining, board.turn, cs.WHITE)
            need_s, need_g = corrected_need_moves_count(
                board, target, avail_s, avail_g, set()
            )
            with self.subTest(depth=depth + 1, usi=usi):
                self.assertLessEqual(need_s, avail_s)
                self.assertLessEqual(need_g, avail_g)


if __name__ == "__main__":
    unittest.main()
