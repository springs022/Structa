import unittest

import cshogi as cs

from board_utils import file_rank_to_sq
from cost_calc import (
    INF,
    corrected_need_moves_count,
    major_p_cost,
    minor_p_cost,
    need_moves_count,
    unprom_move_cost,
)


def sq(file, rank):
    return file_rank_to_sq(file, rank)


class GeometryCostTests(unittest.TestCase):
    def test_unpromoted_piece_costs(self):
        cases = [
            (cs.BPAWN, sq(6, 7), sq(6, 4), 3),
            (cs.WPAWN, sq(6, 3), sq(6, 6), 3),
            (cs.BLANCE, sq(6, 7), sq(6, 2), 1),
            (cs.BROOK, sq(6, 5), sq(2, 3), 2),
            (cs.BBISHOP, sq(4, 3), sq(7, 6), 1),
            (cs.BBISHOP, sq(4, 3), sq(7, 5), 100),
        ]
        for piece, src, dst, expected in cases:
            with self.subTest(piece=piece, src=src, dst=dst):
                self.assertEqual(expected, unprom_move_cost(piece, src, dst))

    def test_promoted_piece_costs(self):
        self.assertEqual(1, minor_p_cost(cs.BPROM_PAWN, sq(5, 6), sq(5, 5)))
        self.assertEqual(1, major_p_cost(cs.BPROM_ROOK, sq(1, 2), sq(1, 7)))
        self.assertEqual(2, major_p_cost(cs.BPROM_ROOK, sq(1, 2), sq(5, 4)))


class PositionCostTests(unittest.TestCase):
    START_SFEN = "lnsgkgsnl/1r7/ppppp2pp/6p2/5p3/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w b 1"
    TARGET_SFEN = "lnsgkgsnl/1r5b1/ppppp2pp/6p2/5p3/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 1"

    def setUp(self):
        self.start = cs.Board(self.START_SFEN)
        self.target = cs.Board(self.TARGET_SFEN)

    def test_need_moves_count_has_stable_totals(self):
        sente, gote = need_moves_count(self.start, self.target)
        self.assertEqual(0, sum(min(x.make_cost, x.move_cost) for x in sente))
        self.assertEqual(1, sum(min(x.make_cost, x.move_cost) for x in gote))

    def test_corrected_cost_accepts_fixed_piece_argument(self):
        self.assertEqual(
            (0, 1),
            corrected_need_moves_count(
                self.start, self.target, avail_s=2, avail_g=1, fixed_rfs=set()
            ),
        )

    def test_corrected_cost_reports_impossible_budget(self):
        self.assertEqual(
            (INF, INF),
            corrected_need_moves_count(
                self.start, self.target, avail_s=0, avail_g=0, fixed_rfs=set()
            ),
        )


if __name__ == "__main__":
    unittest.main()
