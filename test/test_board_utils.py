import unittest

import cshogi as cs

from board_utils import (
    file_rank_to_sq,
    hand_piece_to_board_pieces,
    sq_to_file_rank,
    sq_to_usi,
)


class BoardUtilsTests(unittest.TestCase):
    def test_square_conversions_round_trip(self):
        for file in range(1, 10):
            for rank in range(1, 10):
                sq = file_rank_to_sq(file, rank)
                self.assertEqual((file, rank), sq_to_file_rank(sq))

    def test_square_to_usi(self):
        self.assertEqual("3d", sq_to_usi(file_rank_to_sq(3, 4)))
        self.assertEqual("5a", sq_to_usi(file_rank_to_sq(5, 1)))

    def test_hand_piece_candidates_include_promoted_form_when_possible(self):
        self.assertEqual(
            {cs.BPAWN, cs.BPROM_PAWN},
            hand_piece_to_board_pieces(cs.HPAWN, cs.BLACK),
        )
        self.assertEqual(
            {cs.WROOK, cs.WPROM_ROOK},
            hand_piece_to_board_pieces(cs.HROOK, cs.WHITE),
        )
        self.assertEqual(
            {cs.BGOLD},
            hand_piece_to_board_pieces(cs.HGOLD, cs.BLACK),
        )


if __name__ == "__main__":
    unittest.main()
