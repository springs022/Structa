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
from collections import deque
from typing import Optional, Tuple, Set, List
from dataclasses import dataclass
from board_utils import (
    PROM_PIECES,
    c_distance,
    m_distance,
    sq_to_file_rank,
    file_rank_to_sq,
    is_promoted,
    piece_owner,
    unpromote,
    in_prom_zone,
    change_owner
)
from movement_rules import (
    is_reachable_by_one_move,
    can_move_as_bishop,
    can_move_as_rook,
    can_move_as_prom_rook,
    can_move_as_prom_bishop,
    bishop_attack_sqs
)

@dataclass(frozen=True)
class PieceCost:
    piece: int
    owner: int
    sq: int
    make_cost: int
    move_cost: int

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

def unprom_move_cost(
    src_piece: int,
    src_sq: int,
    dst_sq: int
) -> Optional[int]:
    """
    src_sq にある生駒（src_piece）が dst_sq に生駒のまま到達する最小手数を返す。
    到達不可の場合は100を返す。
    """
    owner = piece_owner(src_piece)
    if owner is None:
        return None
    if is_promoted(src_piece):
        return None
    if src_piece in (cs.BKING, cs.WKING):
        return None
    if src_sq == dst_sq:
        return 0
    
    # 最初に先後を確認し、後手なら先手の駒に置き換える
    if owner == cs.BLACK:
        src_file, src_rank = sq_to_file_rank(src_sq)
        dst_file, dst_rank = sq_to_file_rank(dst_sq)
        piece = src_piece
    else:
        tmp_src_file, tmp_src_rank = sq_to_file_rank(src_sq)
        tmp_dst_file, tmp_dst_rank = sq_to_file_rank(dst_sq)
        src_file = 10 - tmp_src_file
        src_rank = 10 - tmp_src_rank
        dst_file = 10 - tmp_dst_file
        dst_rank = 10 - tmp_dst_rank
        owner = cs.BLACK
        piece = change_owner(src_piece)
    df = dst_file - src_file
    dr = dst_rank - src_rank
    n_src_sq = file_rank_to_sq(src_file, src_rank)
    n_dst_sq = file_rank_to_sq(dst_file, dst_rank)

    # --- 飛 ---
    if piece == cs.BROOK:
        if can_move_as_rook(df, dr):
            return 1
        return 2
    # --- 角 ---
    if piece == cs.BBISHOP:
        if can_move_as_bishop(df, dr):
            return 1
        if (df + dr) % 2 == 0:
            return 2
        return 100
    # --- 金 ---
    if piece == cs.BGOLD:
        return minor_p_distance(n_src_sq, n_dst_sq, owner)
    # --- 銀 ---
    if piece == cs.BSILVER:
        if dr < 0 and abs(dr) >= abs(df):
            return minor_p_distance(n_src_sq, n_dst_sq, owner)
        if (df + dr) % 2 == 0:
            return max(abs(dr), abs(df))
        return max(abs(dr) + 1, abs(df)) + 1
    # --- 桂 ---
    if piece == cs.BKNIGHT:
        if abs(df) == 1 and dr == -2:
            return 1
        if abs(df) in (0, 2) and dr == -4:
            return 2
        if abs(df) in (1, 3) and dr == -6:
            return 3
        return 100
    # --- 香 ---
    if piece == cs.BLANCE:
        if df == 0 and dr < 0:
            return 1
        return 100
    # --- 歩 ---
    if piece == cs.BPAWN:
        if df == 0 and dr < 0:
            return -dr
        return 100
    return 100

def major_p_cost(
    src_piece: int,
    src_sq: int,
    dst_sq: int
) -> Optional[int]:
    """
    src_sq にある角・飛・馬・龍（src_piece）が
    dst_sq に成駒として到達する最小手数を返す。
    """
    owner = piece_owner(src_piece)
    if owner is None:
        return None
    base_piece = unpromote(src_piece)
    if base_piece not in (cs.BBISHOP, cs.WBISHOP, cs.BROOK, cs.WROOK):
        return None
    if is_promoted(src_piece) and src_sq == dst_sq:
        return 0
    src_file, src_rank = sq_to_file_rank(src_sq)
    dst_file, dst_rank = sq_to_file_rank(dst_sq)
    norm_src_rank = src_rank if owner == 0 else 10 - src_rank
    norm_dst_rank = dst_rank if owner == 0 else 10 - dst_rank
    df = dst_file - src_file
    dr = norm_dst_rank - norm_src_rank
    if src_piece in (cs.BPROM_ROOK, cs.WPROM_ROOK):
        # 龍→龍
        if can_move_as_prom_rook(df, dr):
            return 1
        else:
            return 2
    elif src_piece in (cs.BROOK, cs.WROOK):
        # 飛→龍
        if norm_src_rank <= 3 or norm_dst_rank <= 3:
            if can_move_as_rook(df, dr):
                return 1
            else:
                return 2
        else:
            if norm_dst_rank == 4 and abs(df) == 1:
                # 成って斜めに引く
                return 2
            elif df == 0:
                # 真っすぐ引く
                return 2
            else:
                return 3
    elif src_piece in (cs.BPROM_BISHOP, cs.WPROM_BISHOP):
        # 馬→馬
        if can_move_as_prom_bishop(df, dr):
            return 1
        elif ((df + dr) % 2 == 0) or (can_move_as_bishop(df, dr + 1)) or (can_move_as_bishop(df, dr - 1)):
            # 角の動き２回／上下左右１回＋角の動き１回
            return 2
        else:
            return 3
    else:
        # 角→馬
        if norm_src_rank <= 3 or norm_dst_rank <= 3:
            if can_move_as_bishop(df, dr):
                return 1
            elif ((df + dr) % 2 == 0) or (can_move_as_bishop(df, dr + 1)) or (can_move_as_bishop(df, dr - 1)):
                return 2
            else:
                return 3
        else: # 出発マスも到着マスも可成地域ではない場合
            promotable_sqs = set() # 角が1手で到達できる可成地域
            attacked_sqs_by_b = bishop_attack_sqs(src_sq)
            for sq in attacked_sqs_by_b:
                f, r = sq_to_file_rank(sq)
                if in_prom_zone(owner, r):
                    promotable_sqs.add(sq)
            if not promotable_sqs:
                #成るのに２手掛かる場合
                for sq in attacked_sqs_by_b:
                    attacked_sqs2 = bishop_attack_sqs(sq)
                    for sq in attacked_sqs2:
                        f, r = sq_to_file_rank(sq)
                        if in_prom_zone(owner, r):
                            promotable_sqs.add(sq)
                cost = 100
                for sq in promotable_sqs:
                    f, r = sq_to_file_rank(sq)
                    norm_r = r if owner == 0 else 10 - r
                    df1 = dst_file - f
                    dr1 = norm_dst_rank - norm_r
                    if can_move_as_prom_bishop(df1, dr1):
                        tmp = 3
                    elif ((df1 + dr1) % 2 == 0) or (can_move_as_bishop(df1, dr1 + 1)) or (can_move_as_bishop(df1, dr1 - 1)):
                        # 角の動き２回／上下左右１回＋角の動き１回
                        tmp = 4
                    else:
                        tmp = 5
                    cost = min(cost, tmp)
                return cost
            else:
                #１手で成れる
                cost = 100
                for sq in promotable_sqs:
                    f, r = sq_to_file_rank(sq)
                    norm_r = r if owner == 0 else 10 - r
                    df1 = dst_file - f
                    dr1 = norm_dst_rank - norm_r
                    if can_move_as_prom_bishop(df1, dr1):
                        tmp = 2
                    elif ((df1 + dr1) % 2 == 0) or (can_move_as_bishop(df1, dr1 + 1)) or (can_move_as_bishop(df1, dr1 - 1)):
                        # 角の動き２回／上下左右１回＋角の動き１回
                        tmp = 3
                    else:
                        tmp = 4
                    cost = min(cost, tmp)
                return cost

def prom_cost(board: cs.Board, piece: int, dst_sq: int) -> Optional[Tuple[int, int]]:
    """
    board において、piece（成駒）を dst_sq に設置するのに掛かる
    最小手数の組（駒打ちから成駒を作る場合, 既存成駒の移動の場合）を返す。
    """
    if not is_promoted(piece):
        return None
    owner = piece_owner(piece)
    if owner is None:
        return None
    if board.piece(dst_sq) == piece:
        return 0, 0
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
            cost = major_p_cost(p, sq, dst_sq)
            if cost is not None:
                move_cost = min(move_cost, cost)
            continue
        # 小駒
        if is_promoted(p):
            move_cost = min(move_cost, minor_p_distance(sq, dst_sq, owner))
            continue
        waypoint_rank = get_waypoint_rank_for(piece)
        if waypoint_rank is None:
            continue
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
    return make_cost, move_cost

def need_moves_count(
    start_board: cs.Board,
    target_board: cs.Board
) -> Tuple[List[PieceCost], List[PieceCost]]:
    """
    target_board に配置されているが start_board に配置されていない駒たちについて、
    各駒ごとのコスト情報を先後別に返す。
    """
    result = [[], []]  # 0:先手, 1:後手

    bk_cost, wk_cost = kings_required_moves(start_board, target_board)
    k_cost = [bk_cost, wk_cost]

    for sq in range(81):
        p = target_board.piece(sq)
        owner = piece_owner(p)
        if owner not in (0, 1):
            continue
        if start_board.piece(sq) == p:
            continue
        # --- 玉 ---
        if p in (cs.BKING, cs.WKING):
            pc = PieceCost(
                piece=p,
                owner=owner,
                sq=sq,
                make_cost=100,
                move_cost=k_cost[owner],
            )
            result[owner].append(pc)
            continue
        # --- 成駒 ---
        if is_promoted(p):
            res = prom_cost(start_board, p, sq)
            if res is None:
                continue
            make_cost, move_cost = res
            pc = PieceCost(
                piece=p,
                owner=owner,
                sq=sq,
                make_cost=make_cost,
                move_cost=move_cost,
            )
            result[owner].append(pc)
            continue
        # --- 生駒 ---
        # 生駒は「打つ」=1、「動かす」は今回は考えない
        pc = PieceCost(
            piece=p,
            owner=owner,
            sq=sq,
            make_cost=1,
            move_cost=100,
        )
        result[owner].append(pc)
    return result[0], result[1]
