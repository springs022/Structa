import unittest

import cshogi as cs

from parallel import (
    _completed_prefix,
    decide_process_count,
    enumerate_first_moves,
)


class FirstMoveEnumerationTests(unittest.TestCase):
    def test_order_matches_serial_search(self):
        board = cs.Board()
        pairs, total = enumerate_first_moves(board, set())
        expected = sorted(
            [cs.move_to_usi(mv) for mv in board.legal_moves]
        )
        self.assertEqual(total, len(expected))
        self.assertEqual(expected, [u for _i, u in pairs])
        self.assertEqual(list(range(total)), [i for i, _u in pairs])

    def test_fixed_pieces_are_excluded_but_indices_keep_gaps(self):
        board = cs.Board()
        pairs, total = enumerate_first_moves(board, {77})
        usis = [u for _i, u in pairs]
        self.assertNotIn("7g7f", usis)
        self.assertEqual(total, len(list(board.legal_moves)))
        # 除外された初手のぶんだけ添字に欠番ができる
        self.assertLess(len(pairs), total)


class CompletedPrefixTests(unittest.TestCase):
    def test_prefix_stops_at_first_incomplete_index(self):
        indices = [0, 1, 2, 3, 4]
        self.assertEqual(3, _completed_prefix({0, 1, 2, 4}, indices, 5))
        self.assertEqual(5, _completed_prefix({0, 1, 2, 3, 4}, indices, 5))
        self.assertEqual(0, _completed_prefix(set(), indices, 5))

    def test_excluded_indices_count_as_complete(self):
        # 添字 2 は不動駒で除外されたので探索対象に入っていない
        indices = [0, 1, 3, 4]
        self.assertEqual(5, _completed_prefix({0, 1, 3, 4}, indices, 5))
        self.assertEqual(3, _completed_prefix({0, 1}, indices, 5))


class ProcessCountTests(unittest.TestCase):
    def test_explicit_value_is_respected(self):
        self.assertEqual(3, decide_process_count(3))
        self.assertEqual(1, decide_process_count(1))

    def test_auto_returns_at_least_one(self):
        self.assertGreaterEqual(decide_process_count(0), 1)


if __name__ == "__main__":
    unittest.main()
