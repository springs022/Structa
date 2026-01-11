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
    """
    csa = cs.move_to_csa(mv)
    frm = int(csa[0:2])
    to  = int(csa[2:4])
    return frm in fixed_set or to in fixed_set
