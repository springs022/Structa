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
from collections import defaultdict
from typing import Optional, Tuple, Set, List
from dataclasses import dataclass
from board_utils import (
    c_distance,
    m_distance,
    sq_to_file_rank,
    file_rank_to_sq,
    is_promoted,
    piece_owner,
    unpromote,
    in_prom_zone,
    normalize,
    normalize_piece,
    piece_to_hand_piece,
    HAND_TO_PIECE
)
from movement_rules import (
    can_move_as_bishop,
    can_move_as_rook,
    can_move_as_prom_rook,
    can_move_as_prom_bishop,
    can_move_as_lance,
    bishop_attack_sqs
)

INF = 1000
# 「その駒種は持駒にできないので打てない」ことを表す make_cost。
# INF より小さくしてあるのは、加算しても INF を超えないようにするため。
_MAKE_IMPOSSIBLE = 200

SQUARE_NB = 81
PIECE_NB = 31
MOVE_COST_TABLE_SIZE = PIECE_NB * SQUARE_NB * SQUARE_NB
_COST_UNKNOWN = -2
_COST_NONE = -1
_UNPROM_MOVE_COST_TABLE = [_COST_UNKNOWN] * MOVE_COST_TABLE_SIZE
_MINOR_P_COST_TABLE = [_COST_UNKNOWN] * MOVE_COST_TABLE_SIZE
_MAJOR_P_COST_TABLE = [_COST_UNKNOWN] * MOVE_COST_TABLE_SIZE

# 探索中に piece_to_hand_piece() を繰り返し呼ばないための表。
# 駒種マスクは HPAWN〜HROOK の7bitで表す。
_HAND_KIND_BY_PIECE = tuple(piece_to_hand_piece(piece) for piece in range(PIECE_NB))
_HAND_KIND_BIT_BY_PIECE = tuple(
    0 if kind is None else 1 << kind for kind in _HAND_KIND_BY_PIECE
)
_PIECE_OWNER_BY_PIECE = tuple(piece_owner(piece) for piece in range(PIECE_NB))

def _move_cost_table_index(piece: int, src_sq: int, dst_sq: int) -> int:
    return (piece * SQUARE_NB + src_sq) * SQUARE_NB + dst_sq

def _table_cost(value: int) -> Optional[int]:
    return None if value == _COST_NONE else value

@dataclass(frozen=True)
class PieceCost:
    piece: int
    owner: int
    sq: int
    make_cost: int
    move_cost: int

@dataclass(frozen=True)
class TargetRequirement:
    sq: int
    piece: int
    owner: int
    promoted: bool
    make_cost: int
    # 高速化のための事前計算。この目標マスを埋めうる盤上駒の駒種と、
    # 大駒（馬・龍）かどうかのフラグ。
    candidates: tuple = ()
    is_major: bool = False
    # 持駒としての駒種（HPAWN〜HROOK）。玉は None。
    # 「その駒種を打てるか」の判定に使う。
    hand_kind: object = None

@dataclass(frozen=True)
class TargetInfo:
    pieces: tuple[int, ...]
    requirements: tuple[TargetRequirement, ...]
    # 二歩ペナルティの判定が必要かどうか（と金の設置要求があるかどうか）を先後別に持つ。
    # 不要なら protected_sqs の構築自体を省略できる。
    needs_nifu_check: tuple = (False, False)
    # 目標局面での持駒総数（先手, 後手）。駒を取る手の必要数を数えるのに使う。
    hand_totals: tuple = (0, 0)

class BoardAnalysis:
    """
    現局面の解析結果。

    protected_sqs は「目標局面と一致しているマス」の集合だが、
    実際に参照されるのは
        - need == avail のときの再計算（takeable_piece_types）
        - と金の設置要求があるときの二歩ペナルティ
    に限られる。毎ノードで作ると無駄が大きいので遅延生成にしている。
    """

    __slots__ = ("pieces", "piece_positions", "target_pieces", "_protected_sqs")

    def __init__(self, pieces, piece_positions, target_pieces):
        self.pieces = pieces
        self.piece_positions = piece_positions
        self.target_pieces = target_pieces
        self._protected_sqs = None

    @property
    def protected_sqs(self) -> set:
        cached = self._protected_sqs
        if cached is None:
            target_pieces = self.target_pieces
            cached = {
                sq for sq, piece in enumerate(self.pieces)
                if piece != cs.NONE and piece == target_pieces[sq]
            }
            self._protected_sqs = cached
        return cached

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

def _unprom_move_cost_uncached(
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
        return c_distance(src_sq, dst_sq)
    if src_sq == dst_sq:
        return 0
    
    # 後手の駒なら先手視点にする
    piece, src_file, src_rank = normalize(owner, src_piece, src_sq)
    _, dst_file, dst_rank = normalize(owner, piece, dst_sq)
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
        return minor_p_distance(n_src_sq, n_dst_sq, cs.BLACK)
    # --- 銀 ---
    if piece == cs.BSILVER:
        if dr < 0 and abs(dr) >= abs(df):
            return minor_p_distance(n_src_sq, n_dst_sq, cs.BLACK)
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

def unprom_move_cost(
    src_piece: int,
    src_sq: int,
    dst_sq: int
) -> Optional[int]:
    """生駒の最小移動コストを固定長テーブルから返す。"""
    if not (0 <= src_piece < PIECE_NB and 0 <= src_sq < SQUARE_NB and 0 <= dst_sq < SQUARE_NB):
        return _unprom_move_cost_uncached(src_piece, src_sq, dst_sq)
    idx = _move_cost_table_index(src_piece, src_sq, dst_sq)
    cached = _UNPROM_MOVE_COST_TABLE[idx]
    if cached != _COST_UNKNOWN:
        return _table_cost(cached)
    cost = _unprom_move_cost_uncached(src_piece, src_sq, dst_sq)
    _UNPROM_MOVE_COST_TABLE[idx] = _COST_NONE if cost is None else cost
    return cost

def _minor_p_cost_uncached(
    src_piece: int,
    src_sq: int,
    dst_sq: int
) -> Optional[int]:
    """
    src_sq にある銀、桂、香、歩（src_piece）が
    dst_sq に成駒として到達する最小手数を返す。
    """
    owner = piece_owner(src_piece)
    if owner is None:
        return None
    # 後手の駒なら先手視点にする
    piece, src_file, src_rank = normalize(owner, src_piece, src_sq)
    _, dst_file, dst_rank = normalize(owner, piece, dst_sq)
    if unpromote(piece) not in (cs.BSILVER, cs.BKNIGHT, cs.BLANCE, cs.PAWN):
        return None
    df = dst_file - src_file
    dr = dst_rank - src_rank
    n_src_sq = file_rank_to_sq(src_file, src_rank)
    n_dst_sq = file_rank_to_sq(dst_file, dst_rank)
    is_src_in_prom = in_prom_zone(cs.BLACK, src_rank)
    is_dst_in_prom = in_prom_zone(cs.BLACK, dst_rank)
    if is_promoted(piece):
        return minor_p_distance(n_src_sq, n_dst_sq, cs.BLACK)
    move_cost = 100
    # --- 歩 ---
    if piece == cs.BPAWN:
        if src_rank == 1:
            return move_cost
        if is_src_in_prom:
            waypoint = file_rank_to_sq(src_file, src_rank - 1)
        else:
            waypoint = file_rank_to_sq(src_file, 3)
        move_cost = unprom_move_cost(piece, n_src_sq, waypoint) + minor_p_distance(waypoint, n_dst_sq, cs.BLACK)
    # --- 香 ---
    if piece == cs.BLANCE:
        if src_rank == 1:
            return move_cost
        if is_dst_in_prom and can_move_as_lance(cs.BLACK, df, dr):
            move_cost = 1
        else:
            if is_src_in_prom:
                waypoint = file_rank_to_sq(src_file, src_rank - 1)
            else:
                waypoint = file_rank_to_sq(src_file, 3)
            move_cost = unprom_move_cost(piece, n_src_sq, waypoint) + minor_p_distance(waypoint, n_dst_sq, cs.BLACK)
    # --- 桂 ---
    if piece == cs.BKNIGHT:
        if src_rank in (1, 2):
            return move_cost
        f = src_file
        r = src_rank
        first = True
        while r >= 3:
            if not first and r <= 3:
                break
            cand_file1 = f - 1
            cand_file2 = f + 1
            if abs(cand_file1 - dst_file) > abs(cand_file2 - dst_file):
                f = cand_file2
            else:
                f = cand_file1
            r -= 2
            first = False
        waypoint = file_rank_to_sq(f, r)
        move_cost = unprom_move_cost(piece, n_src_sq, waypoint) + minor_p_distance(waypoint, n_dst_sq, cs.BLACK)
    # --- 銀 ---
    if piece == cs.BSILVER:
        if is_src_in_prom:
            if dr <= 0:
                corr = 0
            else:
                # 引きの移動が何回有効か
                corr = min(abs(dr), abs(df), 4 - src_rank)
            move_cost = minor_p_distance(n_src_sq, n_dst_sq, cs.BLACK) - corr
        else:
            if is_dst_in_prom:
                move_cost = minor_p_distance(n_src_sq, n_dst_sq, cs.BLACK)
            else:
                # 3段目に到達するまで指してもn_dst_sqの筋に到達していないなら、1回引き成りが効く
                tmp = src_rank - 3
                if abs(df) <= tmp:
                    move_cost = tmp + dst_rank - 3
                else:
                    if df < 0:
                        w_file = src_file - tmp
                    else:
                        w_file = src_file + tmp
                    waypoint = file_rank_to_sq(w_file, 3)
                    move_cost = tmp + minor_p_distance(waypoint, n_dst_sq, cs.BLACK) - 1
    return move_cost

def minor_p_cost(
    src_piece: int,
    src_sq: int,
    dst_sq: int
) -> Optional[int]:
    """小駒が成駒で到達する最小コストを固定長テーブルから返す。"""
    if not (0 <= src_piece < PIECE_NB and 0 <= src_sq < SQUARE_NB and 0 <= dst_sq < SQUARE_NB):
        return _minor_p_cost_uncached(src_piece, src_sq, dst_sq)
    idx = _move_cost_table_index(src_piece, src_sq, dst_sq)
    cached = _MINOR_P_COST_TABLE[idx]
    if cached != _COST_UNKNOWN:
        return _table_cost(cached)
    cost = _minor_p_cost_uncached(src_piece, src_sq, dst_sq)
    _MINOR_P_COST_TABLE[idx] = _COST_NONE if cost is None else cost
    return cost

def _major_p_cost_uncached(
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

def major_p_cost(
    src_piece: int,
    src_sq: int,
    dst_sq: int
) -> Optional[int]:
    """大駒が成駒で到達する最小コストを固定長テーブルから返す。"""
    if not (0 <= src_piece < PIECE_NB and 0 <= src_sq < SQUARE_NB and 0 <= dst_sq < SQUARE_NB):
        return _major_p_cost_uncached(src_piece, src_sq, dst_sq)
    idx = _move_cost_table_index(src_piece, src_sq, dst_sq)
    cached = _MAJOR_P_COST_TABLE[idx]
    if cached != _COST_UNKNOWN:
        return _table_cost(cached)
    cost = _major_p_cost_uncached(src_piece, src_sq, dst_sq)
    _MAJOR_P_COST_TABLE[idx] = _COST_NONE if cost is None else cost
    return cost

def prom_cost(board: cs.Board, piece: int, dst_sq: int) -> Optional[Tuple[int, int]]:
    """
    board において、piece（成駒）を dst_sq に設置するのに掛かる
    最小手数の組（駒打ちから成駒を作る場合, 盤上駒の移動の場合）を返す。
    """
    if not is_promoted(piece):
        return None
    owner = piece_owner(piece)
    if owner is None:
        return None
    if board.piece(dst_sq) == piece:
        return 0, 0
    base_piece = unpromote(piece)
    candidates = {piece, base_piece}
    _, dst_rank = sq_to_file_rank(dst_sq)
    norm_rank = dst_rank if owner == 0 else 10 - dst_rank
    move_cost = 100
    if piece in (
        cs.BPROM_PAWN, cs.WPROM_PAWN,
        cs.BPROM_LANCE, cs.WPROM_LANCE,
        cs.BPROM_KNIGHT, cs.WPROM_KNIGHT
    ):
        make_cost = max(2, norm_rank - 1)
    elif piece in (
        cs.BPROM_SILVER, cs.WPROM_SILVER
    ):
        make_cost = max(2, norm_rank - 2)
    else:
        # 龍・馬は持駒を打って作るなら必ず２手
        make_cost = 2

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
        cost = minor_p_cost(p, sq, dst_sq)
        if cost is not None:
            move_cost = min(move_cost, cost)
    return make_cost, move_cost

def prom_cost_w_pos(
    board: cs.Board,
    piece: int,
    dst_sq: int,
    piece_positions: dict[int, list[int]],
    precomputed_make_cost: Optional[int] = None,
) -> Optional[Tuple[int, int]]:
    """
    board において、piece（成駒）を dst_sq に設置するのに掛かる
    最小手数の組（駒打ちから成駒を作る場合, 盤上駒の移動の場合）を返す。
    """
    if not is_promoted(piece):
        return None
    owner = piece_owner(piece)
    if owner is None:
        return None
    if board.piece(dst_sq) == piece:
        return 0, 0
    base_piece = unpromote(piece)
    candidates = {piece, base_piece}
    move_cost = 100
    if precomputed_make_cost is not None:
        make_cost = precomputed_make_cost
    else:
        make_cost = target_make_cost(piece, owner, dst_sq, True)
    
    for p in candidates:
        for sq in piece_positions.get(p, ()):
            # 大駒
            if piece in (cs.BPROM_BISHOP, cs.WPROM_BISHOP, cs.BPROM_ROOK, cs.WPROM_ROOK):
                cost = major_p_cost(p, sq, dst_sq)
                if cost is not None:
                    move_cost = min(move_cost, cost)
                continue
            # 小駒
            cost = minor_p_cost(p, sq, dst_sq)
            if cost is not None:
                move_cost = min(move_cost, cost)
    return make_cost, move_cost

def unprom_cost(board: cs.Board, piece: int, dst_sq: int) -> Optional[Tuple[int, int]]:
    """
    board において、piece（生駒）を dst_sq に設置するのに掛かる
    最小手数の組（駒打ちで実現する場合, 既存生駒の移動の場合）を返す。
    """
    if is_promoted(piece):
        return None
    owner = piece_owner(piece)
    if owner is None:
        return None
    if board.piece(dst_sq) == piece:
        return 0, 0
    make_cost = 1
    move_cost = 100
    if piece in (cs.BKING, cs.WKING):
        make_cost = 100
    for sq in range(81):
        p = board.piece(sq)
        if p != piece:
            continue
        cost = unprom_move_cost(p, sq, dst_sq)
        move_cost = min(move_cost, cost)
    return make_cost, move_cost

def unprom_cost_w_pos(
    board: cs.Board,
    piece: int,
    dst_sq: int,
    piece_positions: dict[int, list[int]],
    precomputed_make_cost: Optional[int] = None,
) -> Optional[Tuple[int, int]]:
    """
    board において、piece（生駒）を dst_sq に設置するのに掛かる
    最小手数の組（駒打ちで実現する場合, 既存生駒の移動の場合）を返す。
    """
    if is_promoted(piece):
        return None
    owner = piece_owner(piece)
    if owner is None:
        return None
    if board.piece(dst_sq) == piece:
        return 0, 0
    make_cost = (
        precomputed_make_cost
        if precomputed_make_cost is not None
        else target_make_cost(piece, owner, dst_sq, False)
    )
    move_cost = 100
    for sq in piece_positions.get(piece, ()):
        cost = unprom_move_cost(piece, sq, dst_sq)
        if cost < move_cost:
            move_cost = cost
    return make_cost, move_cost

def build_piece_positions(board: cs.Board) -> dict[int, list[int]]:
    """
    盤上駒の位置辞書を返す。
    例
      positions[cs.BPAWN] == [54, 63, 72, ...]
      positions[cs.BROOK] == [10]
    """
    positions = defaultdict(list)
    for sq in range(81):
        p = board.piece(sq)
        if p != cs.NONE:
            positions[p].append(sq)
    return positions

def target_make_cost(piece: int, owner: int, dst_sq: int, promoted: bool) -> int:
    """目標駒を持駒から作るための、局面に依存しない最低手数を返す。"""
    if not promoted:
        return 100 if piece in (cs.BKING, cs.WKING) else 1
    _, dst_rank = sq_to_file_rank(dst_sq)
    norm_rank = dst_rank if owner == cs.BLACK else 10 - dst_rank
    if piece in (
        cs.BPROM_PAWN, cs.WPROM_PAWN,
        cs.BPROM_LANCE, cs.WPROM_LANCE,
        cs.BPROM_KNIGHT, cs.WPROM_KNIGHT,
    ):
        return max(2, norm_rank - 1)
    if piece in (cs.BPROM_SILVER, cs.WPROM_SILVER):
        return max(2, norm_rank - 2)
    return 2

_MAJOR_PROM_PIECES = (
    cs.BPROM_BISHOP, cs.WPROM_BISHOP, cs.BPROM_ROOK, cs.WPROM_ROOK
)

def build_target_info(target_board: cs.Board) -> TargetInfo:
    """探索中に変化しない目標局面の盤上情報を事前計算する。"""
    pieces = tuple(target_board.piece(sq) for sq in range(81))
    requirements = []
    needs_nifu = [False, False]
    for sq, piece in enumerate(pieces):
        owner = piece_owner(piece)
        if owner in (cs.BLACK, cs.WHITE):
            promoted = is_promoted(piece)
            if promoted:
                candidates = (piece, unpromote(piece))
            else:
                candidates = (piece,)
            if piece in (cs.BPROM_PAWN, cs.WPROM_PAWN):
                needs_nifu[owner] = True
            requirements.append(
                TargetRequirement(
                    sq, piece, owner, promoted,
                    target_make_cost(piece, owner, sq, promoted),
                    candidates,
                    piece in _MAJOR_PROM_PIECES,
                    _HAND_KIND_BY_PIECE[piece],
                )
            )
    target_hands = target_board.pieces_in_hand
    return TargetInfo(
        pieces, tuple(requirements), (needs_nifu[0], needs_nifu[1]),
        (sum(target_hands[0]), sum(target_hands[1])),
    )

def analyze_board(board: cs.Board, target_info: TargetInfo) -> BoardAnalysis:
    """現在局面を1回走査し、コスト計算で共有する情報を構築する。"""
    pieces = board.pieces
    positions = defaultdict(list)
    for sq, piece in enumerate(pieces):
        if piece != cs.NONE:
            positions[piece].append(sq)
    return BoardAnalysis(pieces, positions, target_info.pieces)

def _move_cost_for_requirement(req: TargetRequirement, piece_positions: dict) -> int:
    """
    盤上の駒を動かして req の目標マスを埋める最小手数。

    ここは探索の最内周なので、minor_p_cost / major_p_cost / unprom_move_cost の
    関数呼び出しを避け、事前計算テーブルを直接引く。
    テーブルが未計算（_COST_UNKNOWN）のときだけ元の関数を呼んで埋める。
    """
    dst_sq = req.sq
    move_cost = 100
    if req.promoted:
        table = _MAJOR_P_COST_TABLE if req.is_major else _MINOR_P_COST_TABLE
        cost_fn = major_p_cost if req.is_major else minor_p_cost
        for cand in req.candidates:
            base = cand * SQUARE_NB
            for src_sq in piece_positions.get(cand, ()):
                idx = (base + src_sq) * SQUARE_NB + dst_sq
                c = table[idx]
                if c == _COST_UNKNOWN:
                    cost_fn(cand, src_sq, dst_sq)
                    c = table[idx]
                if 0 <= c < move_cost:
                    move_cost = c
    else:
        piece = req.piece
        base = piece * SQUARE_NB
        table = _UNPROM_MOVE_COST_TABLE
        for src_sq in piece_positions.get(piece, ()):
            idx = (base + src_sq) * SQUARE_NB + dst_sq
            c = table[idx]
            if c == _COST_UNKNOWN:
                unprom_move_cost(piece, src_sq, dst_sq)
                c = table[idx]
            if 0 <= c < move_cost:
                move_cost = c
    return move_cost

def _missing_requirements_within_budget(board_analysis: BoardAnalysis,
                                        target_info: TargetInfo,
                                        avail_s: int,
                                        avail_g: int):
    """
    未達成要求を集める。未達成要求はそれぞれ最低でも1手必要なので、
    その個数が残り手数を超えた時点で None を返す。

    この判定は obtainable_hand_kinds や move_cost の計算より安いため、
    精密な下界計算の前に行う。
    """
    board_pieces = board_analysis.pieces
    missing = []
    count_s = 0
    count_g = 0
    for req in target_info.requirements:
        if board_pieces[req.sq] == req.piece:
            continue
        missing.append(req)
        if req.owner == cs.BLACK:
            count_s += 1
            if count_s > avail_s:
                return None
        else:
            count_g += 1
            if count_g > avail_g:
                return None
    return missing

def _collect_costs(board_analysis: BoardAnalysis,
                   target_info: TargetInfo,
                   avail_s: int,
                   avail_g: int,
                   obtainable=None,
                   missing_requirements=None):
    """
    先後別に (piece, sq, make_cost, move_cost) のタプル列と、
    min(make, move) の総和を返す。

    途中で総和が avail を超えた時点で打ち切り None を返す（早期脱出）。
    呼び出し側は「need > avail か」しか見ないため、この打ち切りで判定は変わらない。

    obtainable を渡すと「その駒種を持駒にできない側」の make_cost を
    到達不能扱いにする（自分の持駒にも無く、相手も 1 枚も持っていない駒種は
    取ることも打つこともできないため）。
    """
    board_pieces = board_analysis.pieces
    piece_positions = board_analysis.piece_positions
    costs_s = []
    costs_g = []
    s_cost = 0
    g_cost = 0
    requirements = (
        target_info.requirements
        if missing_requirements is None
        else missing_requirements
    )
    for req in requirements:
        sq = req.sq
        piece = req.piece
        if missing_requirements is None and board_pieces[sq] == piece:
            continue
        make_cost = req.make_cost
        if obtainable is not None:
            kind = req.hand_kind
            if kind is not None and not (obtainable[req.owner] & (1 << kind)):
                make_cost = _MAKE_IMPOSSIBLE
        move_cost = _move_cost_for_requirement(req, piece_positions)
        base = make_cost if make_cost < move_cost else move_cost
        if req.owner == cs.BLACK:
            s_cost += base
            if s_cost > avail_s:
                return None
            costs_s.append((piece, sq, make_cost, move_cost))
        else:
            g_cost += base
            if g_cost > avail_g:
                return None
            costs_g.append((piece, sq, make_cost, move_cost))
    return costs_s, costs_g, s_cost, g_cost

def obtainable_hand_kinds(piece_positions: dict, hands) -> tuple:
    """
    先後それぞれが「これから持駒にしうる駒種」の7bitマスクを返す。

    持駒にする方法は「自分がすでに持っている」か「相手の駒を取る」しかない。
    相手が盤上にも持駒にも 1 枚も持っていない駒種は、
    どうやっても自分の持駒にできないので、その駒種の目標マスは
    駒打ちでは実現できない（盤上の駒を動かすしかない）。
    """
    hand_s, hand_g = hands
    hand_mask_s = 0
    hand_mask_g = 0
    for kind in range(7):
        bit = 1 << kind
        if hand_s[kind] > 0:
            hand_mask_s |= bit
        if hand_g[kind] > 0:
            hand_mask_g |= bit

    # 盤上駒と持駒を合わせた、現在それぞれの側が所有する駒種。
    owned_s = hand_mask_s
    owned_g = hand_mask_g
    for piece in piece_positions:
        bit = _HAND_KIND_BIT_BY_PIECE[piece]
        if bit == 0:
            continue
        owner = _PIECE_OWNER_BY_PIECE[piece]
        if owner == cs.BLACK:
            owned_s |= bit
        elif owner == cs.WHITE:
            owned_g |= bit

    # 相手が持っている駒種は取れば手に入る。自分の持駒はそのまま打てる。
    obt_s = hand_mask_s | owned_g
    obt_g = hand_mask_g | owned_s
    return obt_s, obt_g

def capture_aware_cost(piece_costs: list,
                       base_cost: int,
                       hand_now_total: int,
                       hand_target_total: int) -> int:
    """
    「駒を打つには、その駒をどこかで取ってこなければならない」ことを
    考慮した必要手数の下界を返す。

    従来の下界は「生駒の設置は打てば 1 手」としていた。
    しかし持駒が空なら、打つ前にその駒を取る手が要る。

    n 個の目標マスを駒打ちで埋めるとすると、持駒の収支から
        取る手の数 = （打つ手の数）＋（目標の持駒数）－（現在の持駒数）
    が成り立つ。打つ手と取る手はどちらもその側の 1 手で、互いに別の手。
    取る手は盤上の移動手なので、駒の移動手数と取る手数は
    「同じ手が両方を兼ねうる」ため max で束ねる（足してはいけない）。

        総手数 ≥ n ＋ max( 移動手数の合計 , 必要な取る手数 )

    n をどう選ぶかは最小化する。ある n に対して移動手数を最小にするには、
    「打ちに切り替えたときの移動手数の減り」が大きい順に n 個選べばよい。

    持駒が潤沢（取る手が不要）なときは従来の下界と完全に一致する。
    どの n でも従来の値以上になるので、下界が弱くなることはない。
    """
    n_req = len(piece_costs)
    if n_req == 0:
        return base_cost
    # 取る手が 1 つも要らないなら従来どおり
    if hand_now_total >= n_req + hand_target_total:
        return base_cost

    total_move = 0
    savings = []
    for _piece, _sq, make_cost, move_cost in piece_costs:
        total_move += move_cost
        # 打ちに切り替えると、移動手数 move_cost が
        # 「打った後の成りなどの手数（make_cost - 1）」に置き換わる
        savings.append(move_cost - (make_cost - 1))
    savings.sort(reverse=True)

    deficit = hand_target_total - hand_now_total
    journey = total_move
    need_capture = deficit if deficit > 0 else 0
    best = journey if journey > need_capture else need_capture
    for n, saving in enumerate(savings, 1):
        journey -= saving
        need_capture = n + deficit
        if need_capture < 0:
            need_capture = 0
        value = n + (journey if journey > need_capture else need_capture)
        if value < best:
            best = value
    return best

def need_moves_count(
    start_board: cs.Board,
    target_board: cs.Board,
    target_info: Optional[TargetInfo] = None,
    board_analysis: Optional[BoardAnalysis] = None,
) -> Tuple[List[PieceCost], List[PieceCost]]:
    """
    target_board に配置されているが start_board に配置されていない駒たちについて、
    各駒ごとのコスト情報を先後別に返す。
    """
    result = [[], []]  # 0:先手, 1:後手
    if target_info is None:
        target_info = build_target_info(target_board)
    if board_analysis is None:
        board_analysis = analyze_board(start_board, target_info)
    piece_positions = board_analysis.piece_positions

    for requirement in target_info.requirements:
        sq = requirement.sq
        p = requirement.piece
        owner = requirement.owner
        if board_analysis.pieces[sq] == p:
            continue
        if requirement.promoted:
            res = prom_cost_w_pos(
                start_board, p, sq, piece_positions, requirement.make_cost
            )
        else:
            res = unprom_cost_w_pos(
                start_board, p, sq, piece_positions, requirement.make_cost
            )
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
    return result[0], result[1]

def nifu_penalty_for_side(
    side: int,
    piece_costs: list,
    start_board: cs.Board,
    protected_sqs: set[int],
    board_pieces: Optional[list[int]] = None,
) -> int:
    """
    二歩に関する必要追加手数を返す。
    と金の設置が必要な状況で、すでに同じ筋に目標達成済の歩があれば、少なくとも１手余計に掛かる。これらの総和。

    piece_costs は PieceCost でも (piece, sq, make_cost, move_cost) タプルでもよい。
    """
    prom_pawn = cs.BPROM_PAWN if side == 0 else cs.WPROM_PAWN
    pawn = cs.BPAWN if side == 0 else cs.WPAWN

    # 設置が必要な「と金」の筋
    prom_pawn_files = set()
    for pc in piece_costs:
        if isinstance(pc, PieceCost):
            pc_piece, pc_sq = pc.piece, pc.sq
        else:
            pc_piece, pc_sq = pc[0], pc[1]
        if pc_piece == prom_pawn:
            f, _ = sq_to_file_rank(pc_sq)
            prom_pawn_files.add(f)

    if not prom_pawn_files:
        return 0

    # 達成済の歩の筋
    protected_pawn_files = set()
    for sq in protected_sqs:
        piece = board_pieces[sq] if board_pieces is not None else start_board.piece(sq)
        if piece == pawn:
            f, _ = sq_to_file_rank(sq)
            protected_pawn_files.add(f)

    return len(prom_pawn_files & protected_pawn_files)

def corrected_need_moves_count(
    start_board: cs.Board,
    target_board: cs.Board,
    avail_s: int,
    avail_g: int,
    fixed_sqs: set[int],
    target_info: Optional[TargetInfo] = None,
    hands=None,
    precise: bool = True,
) -> Tuple[int, int]:
    """
    先後それぞれが目標局面を実現するのに最低限必要な手数を返す。

    fixed_sqs は不動駒の「square index（0〜80）」の集合。
    hands には board.pieces_in_hand を渡せる（呼び出し側で取得済みなら再取得しない）。
    """
    if target_info is None:
        target_info = build_target_info(target_board)
    board_analysis = analyze_board(start_board, target_info)

    missing_requirements = _missing_requirements_within_budget(
        board_analysis, target_info, avail_s, avail_g
    )
    if missing_requirements is None:
        return INF, INF

    obtainable = None
    if precise:
        if hands is None:
            hands = start_board.pieces_in_hand
        obtainable = obtainable_hand_kinds(board_analysis.piece_positions, hands)

    collected = _collect_costs(
        board_analysis, target_info, avail_s, avail_g, obtainable,
        missing_requirements,
    )
    if collected is None:
        return INF, INF
    piece_costs_s, piece_costs_g, s_cost, g_cost = collected

    if precise:
        # 「打つ前に取ってくる手」を数え込んだ下界に引き上げる。
        # 従来の値以上にしかならないので、早期脱出の判定はそのままで正しい。
        hand_target_s, hand_target_g = target_info.hand_totals
        s_cost = capture_aware_cost(
            piece_costs_s, s_cost, sum(hands[0]), hand_target_s
        )
        if s_cost > avail_s:
            return INF, INF
        g_cost = capture_aware_cost(
            piece_costs_g, g_cost, sum(hands[1]), hand_target_g
        )
        if g_cost > avail_g:
            return INF, INF

    untakeable_sqs = None

    # 再計算（取れない駒・打てない駒は make_cost を使えない）は
    # need == avail のときにしか起動しない。前段のコスト集計で
    # avail を超えたものは既に打ち切られているので、ここでの判定は等号のみでよい。
    if s_cost == avail_s or g_cost == avail_g:
        piece_positions = board_analysis.piece_positions
        untakeable_sqs = board_analysis.protected_sqs | fixed_sqs
        if hands is None:
            hands = start_board.pieces_in_hand
        hand_s, hand_g = hands
        droppable_pieces_s = set()
        droppable_pieces_g = set()
        for hand_idx, count in enumerate(hand_s):
            if count > 0 and hand_idx in HAND_TO_PIECE:
                droppable_pieces_s.add(HAND_TO_PIECE[hand_idx])
        for hand_idx, count in enumerate(hand_g):
            if count > 0 and hand_idx in HAND_TO_PIECE:
                droppable_pieces_g.add(HAND_TO_PIECE[hand_idx])

        takeable_cache = [None, None]

        def takeable_piece_types(side: int) -> set[int]:
            cached = takeable_cache[side]
            if cached is not None:
                return cached
            result = set()
            for piece, sqs in piece_positions.items():
                if piece_owner(piece) != side:
                    continue
                for sq in sqs:
                    if sq not in untakeable_sqs:
                        result.add(normalize_piece(piece))
                        break
            takeable_cache[side] = result
            return result

        def recount(piece_costs, takeable_pieces, droppable_pieces,
                    current: int) -> int:
            """
            取れない・打てない駒種は make_cost を使えない、という再計算。
            capture_aware_cost で引き上げた値より小さくなることがあるので、
            必ず大きい方を採る（どちらも下界なので max で束ねてよい）。
            """
            total = 0
            for piece, _sq, make_cost, move_cost in piece_costs:
                np_piece = normalize_piece(piece)
                if np_piece in takeable_pieces or np_piece in droppable_pieces:
                    total += make_cost if make_cost < move_cost else move_cost
                else:
                    total += move_cost
            return total if total > current else current

        # 後手の再計算（1回目）
        if s_cost == avail_s and g_cost <= avail_g:
            g_cost = recount(
                piece_costs_g, takeable_piece_types(cs.BLACK), droppable_pieces_g,
                g_cost
            )
            if g_cost > avail_g:
                return INF, INF
        # 先手の再計算
        if g_cost == avail_g and s_cost <= avail_s:
            s_cost = recount(
                piece_costs_s, takeable_piece_types(cs.WHITE), droppable_pieces_s,
                s_cost
            )
            if s_cost > avail_s:
                return INF, INF
        # 後手の再計算（2回目）
        if s_cost == avail_s and g_cost <= avail_g:
            g_cost = recount(
                piece_costs_g, takeable_piece_types(cs.BLACK), droppable_pieces_g,
                g_cost
            )

    # 二歩の考慮（と金の設置要求がある側だけ）
    needs_nifu_s, needs_nifu_g = target_info.needs_nifu_check
    if needs_nifu_s or needs_nifu_g:
        if untakeable_sqs is None:
            untakeable_sqs = board_analysis.protected_sqs | fixed_sqs
        board_pieces = board_analysis.pieces
        if needs_nifu_s:
            s_cost += nifu_penalty_for_side(
                cs.BLACK, piece_costs_s, start_board, untakeable_sqs, board_pieces
            )
        if needs_nifu_g:
            g_cost += nifu_penalty_for_side(
                cs.WHITE, piece_costs_g, start_board, untakeable_sqs, board_pieces
            )
    return s_cost, g_cost

