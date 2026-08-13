# Structa - Shogi Proof Game Proofer
# Copyright (C) 2026 Masataka Izumi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import cshogi as cs
from io_utils import out
from board_utils import count_pieces

def adjust_target_turn(start_board: cs.Board, target_board: cs.Board, max_depth: int) -> None:
    """
    start_board の手番を基準に、
    max_depth 手後の手番として target_board.turn を補正する
    """
    start_turn = start_board.turn

    if max_depth % 2 == 0:
        expected_turn = start_turn
    else:
        expected_turn = cs.BLACK if start_turn == cs.WHITE else cs.WHITE

    if target_board.turn != expected_turn:
        target_board.turn = expected_turn
        out("手数に合わせて指定局面の手番を変更しました", 2)

def validate_piece_counts(start_board: cs.Board, target_board: cs.Board):
    c1 = count_pieces(start_board)
    c2 = count_pieces(target_board)
    if c1 != c2:
        raise ValueError(
            f"開始局面と指定局面で各駒種の枚数が一致しません:\n"
            f"start : {c1}\n"
            f"target: {c2}"
        )

def validate_sfen_has_king(sfen: str):
    """
    SFEN が盤上に
    ・先手玉(K)をちょうど1枚
    ・後手玉(k)をちょうど1枚
    含んでいるかをチェックする。
    満たさない場合は ValueError を投げる。
    """
    board_part = sfen.split()[0]

    black_king_count = 0
    white_king_count = 0

    for ch in board_part:
        if ch == 'K':
            black_king_count += 1
        elif ch == 'k':
            white_king_count += 1

    if black_king_count != 1 or white_king_count != 1:
        raise ValueError(
            f"双方ちょうど1枚の玉を含む必要があります。:\n"
            f"先手={black_king_count}, 後手={white_king_count}\n"
            f"sfen={sfen}"
        )

def validate_two_digits(x):
    """
    2桁の数字を (a, b) に分解する。
    例: 76 -> (7, 6)
    """
    s = str(x)
    if len(s) != 2 or not s.isdigit():
        raise ValueError(f"2桁の数字ではありません: {x}")
    a = int(s[0])
    b = int(s[1])
    if not (1 <= a <= 9 and 1 <= b <= 9):
        raise ValueError(f"筋段が範囲外です: {x}")
    return a, b

def is_move_touching_fixed_piece(mv: int, fixed_set: set):
    """
    mv：着手
    fixed_set：例 {13, 19, 21}

    戻り値：
        True：mv の移動元または移動先が fixed_set に含まれる
        False：含まれない

    ※ 探索の最内周では文字列化のコストが無視できないため、
      square index を使う is_move_touching_fixed_sqs を使うこと。
      こちらは互換用に残している。
    """
    csa = cs.move_to_csa(mv)
    frm = int(csa[0:2])
    to  = int(csa[2:4])
    return frm in fixed_set or to in fixed_set

def rf_to_sq(rf: int) -> int:
    """2桁の筋段（例 76）を square index（0〜80）に変換する。"""
    return (rf // 10 - 1) * 9 + (rf % 10 - 1)

def rfs_to_sqs(fixed_rfs: set) -> set:
    """2桁筋段の集合を square index の集合に変換する。"""
    return {rf_to_sq(rf) for rf in fixed_rfs}

# cshogi のビット取り出し API。存在しない版のために getattr で拾う。
_move_from = getattr(cs, "move_from", None)
_move_to = getattr(cs, "move_to", None)
_HAS_BIT_API = _move_from is not None and _move_to is not None

def is_move_touching_fixed_sqs(mv: int, fixed_sqs: set) -> bool:
    """
    mv の移動元または移動先が fixed_sqs（square index の集合）に含まれるか。

    駒打ちの場合、cshogi の move_from は 81 以上の値（持駒種）を返すため、
    0〜80 しか持たない fixed_sqs には決して一致しない。
    これは旧実装（CSA の移動元 "00" が 0 になり一致しない）と同じ挙動。
    """
    if _HAS_BIT_API:
        return _move_to(mv) in fixed_sqs or _move_from(mv) in fixed_sqs
    csa = cs.move_to_csa(mv)
    frm = int(csa[0:2])
    to = int(csa[2:4])
    return (
        (to and rf_to_sq(to) in fixed_sqs)
        or (frm and rf_to_sq(frm) in fixed_sqs)
    )
