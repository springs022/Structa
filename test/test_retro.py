import unittest

import cshogi as cs

from board_utils import position_key
from retro import (
    build_frontier,
    build_table,
    from_squares,
    previous_positions,
    resolve_sequence,
)


def board_after(usis):
    board = cs.Board()
    for usi in usis:
        board.push_usi(usi)
    return board


class FromSquaresTests(unittest.TestCase):
    def test_black_pawn_comes_from_one_square_behind(self):
        pieces = [cs.NONE] * 81
        # 5五(sq=(5-1)*9+(5-1)=40) に来た先手歩の出発マスは 5六(=41)
        self.assertEqual([41], from_squares(cs.BPAWN, 40, pieces))

    def test_white_pawn_direction_is_mirrored(self):
        pieces = [cs.NONE] * 81
        self.assertEqual([39], from_squares(cs.WPAWN, 40, pieces))

    def test_slider_stops_at_first_occupied_square(self):
        pieces = [cs.NONE] * 81
        # 5五の飛車。5三(=38) に駒を置くと 5二・5一 へは届かない
        pieces[38] = cs.BPAWN
        result = set(from_squares(cs.BROOK, 40, pieces))
        self.assertIn(39, result)          # 5四
        self.assertNotIn(38, result)       # 塞いでいるマス自体は出発マスになれない
        self.assertNotIn(37, result)       # その先
        self.assertIn(41, result)          # 5六（反対方向）

    def test_occupied_square_is_never_a_from_square(self):
        pieces = [cs.NONE] * 81
        pieces[41] = cs.BGOLD
        self.assertEqual([], from_squares(cs.BPAWN, 40, pieces))


class PreviousPositionsTests(unittest.TestCase):
    def test_every_predecessor_actually_reaches_the_position(self):
        board = board_after(["7g7f", "3c3d", "8h2b+"])
        key = position_key(board.sfen())
        count = 0
        for prev, usi in previous_positions(board):
            count += 1
            found = None
            for mv in prev.legal_moves:
                if cs.move_to_usi(mv) == usi:
                    found = mv
                    break
            self.assertIsNotNone(found, f"{usi} が prev の合法手に無い")
            prev.push(found)
            self.assertEqual(key, position_key(prev.sfen()))
        self.assertGreater(count, 0)

    def test_actual_predecessor_is_enumerated(self):
        moves = ["7g7f", "3c3d", "8h2b+"]
        board = board_after(moves)
        expected = position_key(board_after(moves[:-1]).sfen())
        keys = {position_key(prev.sfen()) for prev, _usi in previous_positions(board)}
        self.assertIn(expected, keys)

    def test_promotion_predecessor_keeps_the_promotion_flag(self):
        # 8八角が 2二で成った直後の局面
        moves = ["7g7f", "3c3d", "8h2b+"]
        board = board_after(moves)
        usis = {
            usi for prev, usi in previous_positions(board)
            if position_key(prev.sfen()) == position_key(board_after(moves[:-1]).sfen())
        }
        self.assertIn("8h2b+", usis)

    def test_already_promoted_piece_moving_is_enumerated(self):
        """到達マスが成駒のとき「すでに成っていた駒が動いた」場合も拾えること。

        2二の馬が 2一へ上下左右1歩で動く。角にはできない動きなので、
        生駒の動きだけを見ていると出発マスを取りこぼす。
        """
        moves = ["7g7f", "3c3d", "8h2b+", "2a3c", "2b2a"]
        board = board_after(moves)
        expected = position_key(board_after(moves[:-1]).sfen())
        found = [
            usi for prev, usi in previous_positions(board)
            if position_key(prev.sfen()) == expected
        ]
        self.assertIn("2b2a", found)
        # 成る手ではないので "+" は付かない
        self.assertNotIn("2b2a+", found)

    def test_promoted_and_unpromoted_origins_are_both_enumerated(self):
        """成駒の到達マスでは「成って来た」「成駒のまま来た」の両方を展開すること。"""
        moves = ["7g7f", "3c3d", "8h2b+", "2a3c", "2b2a"]
        board = board_after(moves)
        usis = {usi for _prev, usi in previous_positions(board)}
        # 2一の馬について、成って来た手（3三など生駒の角の動き＋成り）と
        # 成駒のまま来た手（2二など馬の動き）の両系統が候補に含まれる
        self.assertTrue(
            any(u.endswith("2a+") for u in usis),
            f"成って 2一に来た候補が無い: {sorted(usis)}",
        )
        self.assertTrue(
            any(u.endswith("2a") and not u.endswith("+") and "*" not in u
                for u in usis),
            f"成駒のまま 2一に来た候補が無い: {sorted(usis)}",
        )

    def test_drop_predecessor_is_enumerated(self):
        moves = ["7g7f", "3c3d", "8h2b+", "2a3c", "B*4e"]
        board = board_after(moves)
        expected = position_key(board_after(moves[:-1]).sfen())
        found = [
            usi for prev, usi in previous_positions(board)
            if position_key(prev.sfen()) == expected
        ]
        self.assertIn("B*4e", found)


class FrontierTests(unittest.TestCase):
    SEQUENCES = [
        ["7g7f", "3c3d", "8h2b+", "2a3c"],
        ["2g2f", "8c8d", "2f2e", "8d8e"],
        ["7g7f", "3c3d", "8h2b+", "3a2b"],
    ]

    def test_position_k_plies_before_target_is_in_the_frontier(self):
        for moves in self.SEQUENCES:
            for k in (1, 2):
                with self.subTest(moves=moves, k=k):
                    target = board_after(moves)
                    table, _sizes = build_table(target, k)
                    mid = board_after(moves[: len(moves) - k])
                    h = mid.zobrist_hash() & 0xFFFFFFFFFFFFFFFF
                    self.assertIn(h, table)
                    expected = tuple(moves[len(moves) - k:])
                    self.assertIn(expected, table[h])

    def test_stored_sequences_all_reach_the_target(self):
        moves = ["7g7f", "3c3d", "8h2b+", "2a3c"]
        target = board_after(moves)
        target_key = position_key(target.sfen())
        table, _sizes = build_table(target, 1)
        # 表に載っている局面から手順を指すと必ず目標局面になる
        checked = 0
        for prev, usi in previous_positions(target):
            h = prev.zobrist_hash() & 0xFFFFFFFFFFFFFFFF
            self.assertIn(h, table)
            for seq in table[h]:
                self.assertIsNotNone(
                    resolve_sequence(prev, seq, target_key),
                    f"{seq} が {position_key(prev.sfen())} から目標局面に到達しない",
                )
                checked += 1
        self.assertGreater(checked, 0)

    def test_resolve_sequence_rejects_a_wrong_sequence(self):
        moves = ["7g7f", "3c3d"]
        target = board_after(moves)
        target_key = position_key(target.sfen())
        mid = board_after(moves[:1])
        self.assertIsNotNone(resolve_sequence(mid, ("3c3d",), target_key))
        self.assertIsNone(resolve_sequence(mid, ("8c8d",), target_key))
        self.assertIsNone(resolve_sequence(mid, ("9i9h",), target_key))


class FrontierSelectionTests(unittest.TestCase):
    def test_k_is_capped_so_that_depth_one_frames_remain(self):
        target = board_after(["7g7f", "3c3d", "8h2b+", "2a3c"])
        self.assertIsNone(build_frontier(target, 2, 2))
        self.assertEqual(1, build_frontier(target, 3, 2).k)
        self.assertEqual(2, build_frontier(target, 4, 2).k)
        self.assertEqual(1, build_frontier(target, 10, 1).k)
        self.assertIsNone(build_frontier(target, 10, 0))

    def test_payload_round_trip(self):
        from retro import RetroFrontier

        target = board_after(["7g7f", "3c3d", "8h2b+", "2a3c"])
        frontier = build_frontier(target, 5, 2)
        restored = RetroFrontier.from_payload(frontier.payload())
        self.assertEqual(frontier.k, restored.k)
        self.assertEqual(frontier.table, restored.table)


if __name__ == "__main__":
    unittest.main()
