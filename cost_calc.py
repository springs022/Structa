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
from typing import List
from typing import Tuple
from typing import Optional
from board_utils import (
    PROM_PIECES,
    c_distance,
    m_distance,
    sq_to_file_rank,
    file_rank_to_sq,
    is_promoted,
    piece_owner,
    unpromote,
)
from movement_rules import (
    is_reachable_by_one_move
)

def count_position_diffs(board: cs.Board, target: cs.Board) -> List[int]:
    """
    board と target の局面差異、持駒の差異を計算する。
    """
    diff = 0
    diff_hand_s = 0
    diff_hand_g = 0

    # --- 盤上の差異 ---
    for sq in range(81):
        p1 = board.piece(sq)
        p2 = target.piece(sq)
        if p1 != p2:
            diff += 1

    # --- 持駒の差異 ---
    for color in (0, 1):          # 0 = 先手, 1 = 後手
        for piece in range(0, 6): # 歩香桂銀金角飛
            d = abs(board.pieces_in_hand[color][piece] - target.pieces_in_hand[color][piece])
            diff += d
            if color == 0:
                diff_hand_s += d
            else:
                diff_hand_g += d

    return diff, diff_hand_s, diff_hand_g

def get_king_square(board: cs.Board, color: int) -> int:
    """
    color = 0 → 先手、1 → 後手
    玉が存在すれば sq（0–80）を返す
    存在しなければ None
    """
    target_king = cs.BKING if color == 0 else cs.WKING
    for sq in range(81):
        if board.piece(sq) == target_king:
            return sq
    return None

def available_moves_for_side(remaining_moves: int, next_to_move: int, side: int) -> int:
    """
    残り手数が remaining_moves、
    次の手番が next_to_move (0=先手,1=後手) のとき、
    side (0 or 1) の残り手数を返す。
    """
    if remaining_moves <= 0:
        return 0
    if next_to_move == side:
        return (remaining_moves + 1) // 2
    else:
        return remaining_moves // 2

def kings_required_moves(board: cs.Board, target: cs.Board) -> tuple:
    """
    board 上の双方の玉が target の玉の位置に到達するのに必要な最小手数を返す。
    """
    s_src = get_king_square(board, 0)
    s_dst = get_king_square(target, 0)
    g_src = get_king_square(board, 1)
    g_dst = get_king_square(target, 1)
    if s_src is None or s_dst is None:
        need_s = 0
    else:
        need_s = c_distance(s_src, s_dst)
    if g_src is None or g_dst is None:
        need_g = 0
    else:
        need_g = c_distance(g_src, g_dst)
    return need_s, need_g

def minor_p_distance(src_sq: int, dst_sq: int, owner: int) -> int:
    """
    src_sq にある小駒成駒が dst_sq に到達するまでに掛かる最小手数を返す。
    """
    f1, r1 = sq_to_file_rank(src_sq)
    f2, r2 = sq_to_file_rank(dst_sq)
    if owner == 0:
        use_c = (r1 > r2)
    else:
        use_c = (r2 > r1)
    if use_c:
        return c_distance(src_sq, dst_sq)
    else:
        return m_distance(src_sq, dst_sq)

def prom_cost(board: cs.Board, piece: int, dst_sq: int) -> Optional[int]:
    """
    board において、piece（成駒）を dst_sq に設置するのに掛かる最小手数を返す。
    """
    if not is_promoted(piece):
        return None
    owner = piece_owner(piece)
    if owner is None:
        return None
    if board.piece(dst_sq) == piece:
        return 0
    if piece not in PROM_PIECES:
        return None
    base_piece = unpromote(piece)
    candidates = {piece, base_piece}
    dst_file, dst_rank = sq_to_file_rank(dst_sq)
    norm_rank = dst_rank if owner == 0 else 10 - dst_rank
    move_cost = 100
    if piece in (
        cs.BPROM_PAWN, cs.WPROM_PAWN,
        cs.BPROM_LANCE, cs.WPROM_LANCE,
        cs.BPROM_KNIGHT, cs.WPROM_KNIGHT
    ):
        base_make_cost = max(2, norm_rank - 1)
    elif piece in (
        cs.BPROM_SILVER, cs.WPROM_SILVER
    ):
        base_make_cost = max(2, norm_rank - 2)
    else:
        base_make_cost = 2
    make_cost = base_make_cost

    def get_waypoint_rank_for(piece: int) -> Optional[int]:
        if piece in (
            cs.BPROM_PAWN, cs.WPROM_PAWN,
            cs.BPROM_LANCE, cs.WPROM_LANCE,
            cs.BPROM_KNIGHT, cs.WPROM_KNIGHT
        ):
            if norm_rank <= 3:
                return dst_rank
            return 3 if owner == 0 else 7
        if piece in (
            cs.BPROM_SILVER, cs.WPROM_SILVER
        ):
            if norm_rank <= 3:
                return dst_rank
            return 4 if owner == 0 else 6
        else:
            return None

    for sq in range(81):
        p = board.piece(sq)
        if p not in candidates:
            continue
        # 大駒
        if piece in (cs.BPROM_BISHOP, cs.WPROM_BISHOP, cs.BPROM_ROOK, cs.WPROM_ROOK):
            if is_reachable_by_one_move(board, sq, dst_sq, p):
                move_cost = min(move_cost, 1)
            else:
                move_cost = min(move_cost, 2) # 概算
            continue
        # 小駒
        if is_promoted(p):
            #成駒の移動
            move_cost = min(move_cost, minor_p_distance(sq, dst_sq, owner))
            continue
        # 小駒成駒を作って移動
        waypoint_rank = get_waypoint_rank_for(piece)
        waypoint = file_rank_to_sq(dst_file, waypoint_rank)
        if not is_reachable_by_one_move(board, sq, waypoint, piece):
            continue
        if piece in (
            cs.BPROM_PAWN, cs.WPROM_PAWN,
            cs.BPROM_LANCE, cs.WPROM_LANCE,
            cs.BPROM_KNIGHT, cs.WPROM_KNIGHT,
            cs.BPROM_SILVER, cs.WPROM_SILVER
        ):
            make_cost = min(make_cost, base_make_cost - 1)
    return min(move_cost, make_cost)

def need_moves_count(start_board: cs.Board, target_board: cs.Board) -> Tuple[int, int]:
    """
    target_board に配置されているが start_board に配置されていない駒たちについて、
    各駒に関する最小手数の和を先後それぞれ返す。
    返り値: (先手の必要手数, 後手の必要手数)
    """
    cost_s = 0
    cost_g = 0
    bk_cost, wk_cost = kings_required_moves(start_board, target_board)

    for sq in range(81):
        p = target_board.piece(sq)
        owner = piece_owner(p)
        if owner not in (0, 1):
            continue
        if start_board.piece(sq) == p:
            continue
        # --- 玉 ---
        if p == cs.BKING:
            cost_s += bk_cost
            continue
        elif p == cs.WKING:
            cost_g += wk_cost
            continue
        # --- 成駒 ---
        elif is_promoted(p):
            if owner == 0:
                cost_s += prom_cost(start_board, p, sq)
            else:
                cost_g += prom_cost(start_board, p, sq)
            continue
        # --- 生駒 ---
        else:
            if owner == 0:
                cost_s += 1
            else:
                cost_g += 1
    return cost_s, cost_g

