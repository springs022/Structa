import unittest

import cshogi as cs

from cost_calc import (
    available_moves_for_side,
    capture_aware_cost,
    corrected_need_moves_count,
    obtainable_hand_kinds,
)


def push_legal(board, usis):
    """合法手であることを確かめながら手順を進める。"""
    for usi in usis:
        mv = board.move_from_usi(usi)
        if not board.is_legal(mv):
            raise AssertionError(f"非合法手です: {usi} / {board.sfen()}")
        board.push(mv)
    return board


def costs(*pairs):
    """(make_cost, move_cost) の並びから piece_costs 形式のリストを作る。"""
    return [(cs.BPAWN, 0, make, move) for make, move in pairs]


def base_of(piece_costs):
    return sum(min(mk, mv) for _p, _sq, mk, mv in piece_costs)


class CaptureAwareCostTests(unittest.TestCase):
    def test_empty_requirements(self):
        self.assertEqual(0, capture_aware_cost([], 0, 0, 0))

    def test_all_drops_need_a_capture_each(self):
        # 持駒なし。3マスをすべて打ちで埋めるなら 3 回打って 3 回取る必要がある
        pc = costs((1, 5), (1, 5), (1, 5))
        self.assertEqual(3, base_of(pc))
        self.assertEqual(6, capture_aware_cost(pc, base_of(pc), 0, 0))

    def test_mixed_choice_is_minimised(self):
        # 1マスは動かした方が安い。打ち 2 回＋取り 2 回、移動 1 手は取りと兼ねうる
        pc = costs((1, 1), (1, 5), (1, 5))
        self.assertEqual(3, base_of(pc))
        self.assertEqual(4, capture_aware_cost(pc, base_of(pc), 0, 0))

    def test_ample_hand_reproduces_the_old_bound(self):
        for pc in (
            costs((1, 5), (1, 5), (1, 5)),
            costs((1, 1), (1, 5), (1, 5)),
            costs((3, 6), (2, 2)),
        ):
            with self.subTest(pc=pc):
                base = base_of(pc)
                self.assertEqual(
                    base, capture_aware_cost(pc, base, len(pc), 0)
                )

    def test_target_hand_surplus_forces_captures_even_without_drops(self):
        # 目標局面で持駒が 2 枚必要なのに今 0 枚 → 最低 2 回は取らないといけない
        pc = costs((1, 1),)
        base = base_of(pc)
        self.assertEqual(1, base)
        self.assertEqual(2, capture_aware_cost(pc, base, 0, 2))

    def test_never_weaker_than_the_old_bound(self):
        import itertools

        for combo in itertools.product(
            [(1, 1), (1, 3), (1, 100), (2, 2), (3, 6), (100, 4)], repeat=3
        ):
            pc = costs(*combo)
            base = base_of(pc)
            for hand_now in range(0, 4):
                for hand_target in range(0, 3):
                    with self.subTest(combo=combo, h=hand_now, ht=hand_target):
                        self.assertGreaterEqual(
                            capture_aware_cost(pc, base, hand_now, hand_target),
                            base,
                        )


class ObtainableKindsTests(unittest.TestCase):
    def test_initial_position_everything_is_obtainable(self):
        board = cs.Board()
        obt_s, obt_g = obtainable_hand_kinds(
            {board.piece(sq): [sq] for sq in range(81) if board.piece(sq)},
            board.pieces_in_hand,
        )
        # 初形はどの駒種も相手が持っているので、取れば手に入る
        for kind in (cs.HPAWN, cs.HLANCE, cs.HKNIGHT, cs.HSILVER,
                     cs.HGOLD, cs.HBISHOP, cs.HROOK):
            self.assertIn(kind, obt_s)
            self.assertIn(kind, obt_g)

    def test_kind_owned_only_by_myself_is_not_obtainable(self):
        # 後手の角が盤上にも持駒にも無い局面。先手は角を打てない
        board = cs.Board(
            "lnsgkgsnl/1r7/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        )
        positions = {}
        for sq in range(81):
            p = board.piece(sq)
            if p:
                positions.setdefault(p, []).append(sq)
        obt_s, obt_g = obtainable_hand_kinds(positions, board.pieces_in_hand)
        self.assertNotIn(cs.HBISHOP, obt_s)
        self.assertIn(cs.HBISHOP, obt_g)


class AdmissibilityTests(unittest.TestCase):
    """
    合法な手順のどの途中局面も枝刈りされないこと（下界が過大でないこと）。

    ここが落ちたら下界が不健全で、解を取りこぼす。
    """

    SEQUENCES = {
        # 取りも打ちも無い、持駒が空のまま進む手順。
        # 精密化した下界がいちばん強く効くケースなので、過大評価が出るならここ。
        "pawn_pushes": [
            "7g7f", "3c3d", "2g2f", "8c8d",
            "6g6f", "4c4d", "5g5f", "6c6d",
        ],
        # 飛車先・角道の交換。双方が歩を持駒にし、最後に打つ。
        "exchanges_and_drop": [
            "2g2f", "8c8d", "2f2e", "8d8e", "2e2d", "2c2d", "2h2d",
            "8e8f", "8g8f", "8b8f", "P*2c",
        ],
        # 成りと駒取りを含む手順。
        "promotion": [
            "7g7f", "3c3d", "8h3c+", "2a3c", "2g2f", "8b3b", "2f2e",
        ],
    }

    def _check(self, moves, precise):
        start = cs.Board()
        target = push_legal(start.copy(), moves)
        board = start.copy()
        for depth, usi in enumerate(moves):
            board.push_usi(usi)
            remaining = len(moves) - depth - 1
            avail_s = available_moves_for_side(remaining, board.turn, cs.BLACK)
            avail_g = available_moves_for_side(remaining, board.turn, cs.WHITE)
            need_s, need_g = corrected_need_moves_count(
                board, target, avail_s, avail_g, set(), precise=precise
            )
            with self.subTest(depth=depth + 1, usi=usi, precise=precise):
                self.assertLessEqual(need_s, avail_s)
                self.assertLessEqual(need_g, avail_g)

    def test_legal_sequences_are_never_pruned(self):
        for name, moves in self.SEQUENCES.items():
            for precise in (False, True):
                with self.subTest(sequence=name, precise=precise):
                    self._check(moves, precise)

    def test_precise_bound_is_never_below_the_plain_bound(self):
        for name, moves in self.SEQUENCES.items():
            start = cs.Board()
            target = push_legal(start.copy(), moves)
            board = start.copy()
            for depth, usi in enumerate(moves):
                board.push_usi(usi)
                remaining = len(moves) - depth - 1
                # 予算を十分に取って INF 打ち切りを避け、素の値どうしを比べる
                plain = corrected_need_moves_count(
                    board, target, 99, 99, set(), precise=False
                )
                fine = corrected_need_moves_count(
                    board, target, 99, 99, set(), precise=True
                )
                with self.subTest(sequence=name, depth=depth + 1):
                    self.assertGreaterEqual(fine[0], plain[0])
                    self.assertGreaterEqual(fine[1], plain[1])


if __name__ == "__main__":
    unittest.main()
