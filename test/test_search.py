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
            "search.is_move_touching_fixed_sqs",
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
        # 旧版の手順 ["7g7f","3c3d","8h3c+","2a3c","B*4e","8b7b","4e3d+"] は
        # 実際には合法手順ではなかった。
        #   ・B*4e … この時点で先手は角を持っていない
        #             （8h3c+ は 3c が空きマスなので駒を取っていない）
        #   ・4e3d+ … 4五も3四も敵陣ではないので成れない
        # 到達できない局面で下界を検査していたことになるため、
        # 成りと駒取りを含む合法手順に差し替えた。
        moves = ["7g7f", "3c3d", "8h3c+", "2a3c", "2g2f", "8b3b", "2f2e"]
        target = start.copy()
        for usi in moves:
            mv = target.move_from_usi(usi)
            self.assertTrue(target.is_legal(mv), f"非合法手です: {usi}")
            target.push(mv)

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


class BoardStateTests(unittest.TestCase):
    def test_start_board_is_restored_after_search(self):
        start = cs.Board()
        before = start.sfen()
        target = start.copy()
        target.push_usi("7g7f")

        with contextlib.redirect_stdout(io.StringIO()):
            find_all_paths_to_target(
                start, target, 1, 1, set(), 1, 0, 0, [], []
            )

        # 解数上限に達して break した場合でも開始局面に戻っていること
        self.assertEqual(before, start.sfen())

    def test_two_ply_solution_is_found(self):
        start = cs.Board()
        target = start.copy()
        target.push_usi("7g7f")
        target.push_usi("3c3d")

        with contextlib.redirect_stdout(io.StringIO()):
            solutions, _stats, _idx, _interrupted = find_all_paths_to_target(
                start, target, 2, 1, set(), 4, 0, 0, [], []
            )

        self.assertEqual(1, len(solutions))
        self.assertEqual(
            ["7g7f", "3c3d"], [cs.move_to_usi(mv) for mv in solutions[0]]
        )


class RetroFrontierSearchTests(unittest.TestCase):
    """終端フロンティアの有無で結果が変わらないことを確かめる。"""

    MOVES = ["7g7f", "3c3d", "2g2f", "8c8d"]
    # 先手の2手・後手の2手はそれぞれ順序を入れ替えられるので解は4つ
    EXPECTED = sorted([
        ("7g7f", "3c3d", "2g2f", "8c8d"),
        ("7g7f", "8c8d", "2g2f", "3c3d"),
        ("2g2f", "3c3d", "7g7f", "8c8d"),
        ("2g2f", "8c8d", "7g7f", "3c3d"),
    ])

    def _run(self, retro_plies, moves=None, limit=10):
        moves = moves or self.MOVES
        start = cs.Board()
        target = start.copy()
        for usi in moves:
            target.push_usi(usi)
        with contextlib.redirect_stdout(io.StringIO()):
            solutions, stats, _idx, _interrupted = find_all_paths_to_target(
                start, target, len(moves), limit, set(), 8, 0, 0, [], [],
                retro_plies=retro_plies,
            )
        found = sorted(
            tuple(cs.move_to_usi(mv) for mv in sol) for sol in solutions
        )
        return found, stats

    def test_solutions_match_the_known_answer(self):
        for retro_plies in (0, 1, 2):
            with self.subTest(retro_plies=retro_plies):
                found, stats = self._run(retro_plies)
                self.assertEqual(self.EXPECTED, found)
                self.assertEqual(retro_plies, stats["retro_k"])

    def test_odd_depth_uses_one_ply_frontier(self):
        moves = ["7g7f", "3c3d", "8h2b+"]
        base, _ = self._run(0, moves)
        got, stats = self._run(2, moves)
        self.assertEqual(base, got)
        self.assertEqual(1, stats["retro_k"])   # max_depth-2 で頭打ち
        self.assertIn(("7g7f", "3c3d", "8h2b+"), got)

    def test_frontier_cuts_node_count(self):
        _base, base_stats = self._run(0)
        _got, retro_stats = self._run(2)
        self.assertLess(retro_stats["total_nodes"], base_stats["total_nodes"])
        self.assertGreater(retro_stats["frontier_misses"], 0)

    def test_start_board_is_restored_with_frontier(self):
        start = cs.Board()
        before = start.sfen()
        target = start.copy()
        for usi in self.MOVES:
            target.push_usi(usi)
        with contextlib.redirect_stdout(io.StringIO()):
            find_all_paths_to_target(
                start, target, 4, 10, set(), 8, 0, 0, [], [], retro_plies=2
            )
        self.assertEqual(before, start.sfen())

    def test_fixed_pieces_are_honoured_inside_the_frontier(self):
        # 7七の歩を不動駒にすると 7g7f を含む解は出ない
        start = cs.Board()
        target = start.copy()
        for usi in self.MOVES:
            target.push_usi(usi)
        with contextlib.redirect_stdout(io.StringIO()):
            solutions, _stats, _idx, _interrupted = find_all_paths_to_target(
                start, target, 4, 10, {77}, 8, 0, 0, [], [], retro_plies=2
            )
        self.assertEqual([], solutions)


class PreciseLowerBoundSearchTests(unittest.TestCase):
    """下界の精密化が常に有効で、解を保つことを確かめる。"""

    _run = RetroFrontierSearchTests._run
    MOVES = RetroFrontierSearchTests.MOVES
    EXPECTED = RetroFrontierSearchTests.EXPECTED

    def test_precise_bound_is_always_enabled(self):
        for retro_plies in (0, 2):
            with self.subTest(retro_plies=retro_plies):
                found, stats = self._run(retro_plies)
                self.assertEqual(self.EXPECTED, found)
                self.assertTrue(stats["precise_lb"])


class FixedPieceFilterTests(unittest.TestCase):
    def test_square_based_filter_matches_legacy_filter(self):
        from validation import (
            is_move_touching_fixed_piece,
            is_move_touching_fixed_sqs,
            rfs_to_sqs,
        )

        # 駒打ちを含む局面で比較する（README の作品 3 の目標局面）
        board = cs.Board(
            "lnsgkgsnl/1r7/pppp+Bp1pp/6p2/9/9/PP1P+bPPPP/7R1/LN1GKGSNL b P2ps 1"
        )
        fixed_rfs = {13, 19, 77, 28, 55}
        fixed_sqs = rfs_to_sqs(fixed_rfs)
        for mv in board.legal_moves:
            with self.subTest(usi=cs.move_to_usi(mv)):
                self.assertEqual(
                    is_move_touching_fixed_piece(mv, fixed_rfs),
                    is_move_touching_fixed_sqs(mv, fixed_sqs),
                )

    def test_rf_to_sq_matches_board_utils(self):
        from board_utils import file_rank_to_sq
        from validation import rf_to_sq

        for file in range(1, 10):
            for rank in range(1, 10):
                self.assertEqual(
                    file_rank_to_sq(file, rank), rf_to_sq(file * 10 + rank)
                )


if __name__ == "__main__":
    unittest.main()
