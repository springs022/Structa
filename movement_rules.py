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
from board_utils import (
    in_prom_zone,
    piece_owner,
    unpromote,
    sq_to_file_rank,
    file_rank_to_sq,
    sq_to_usi,
    hand_piece_to_board_pieces,
    is_dead_end_piece,
    piece_to_hand_piece,
    is_check_for_other_side,
    demote,
    are_kings_adjacent,
    has_nifu,
    HAND_PIECE_TO_USI,
    PROM_PIECES
)

INF = 10**9

def can_promote_on_move(owner: int, src_rank: int, dst_rank: int) -> bool:
    return (
        in_prom_zone(owner, src_rank) or
        in_prom_zone(owner, dst_rank)
    )

def can_move_as_gold(owner: int, df: int, dr: int) -> bool:
    if owner == 0:
        return (
            (dr == -1 and abs(df) <= 1) or
            (dr == 0 and abs(df) == 1) or
            (dr == 1 and df == 0)
        )
    else:
        return (
            (dr == 1 and abs(df) <= 1) or
            (dr == 0 and abs(df) == 1) or
            (dr == -1 and df == 0)
        )

def can_move_as_silver(owner: int, df: int, dr: int) -> bool:
    if owner == 0:
        return (
            (dr == -1 and abs(df) <= 1) or
            (dr == 1 and abs(df) == 1)
        )
    else:
        return (
            (dr == 1 and abs(df) <= 1) or
            (dr == -1 and abs(df) == 1)
        )

def can_move_as_knight(owner: int, df: int, dr: int) -> bool:
    return (abs(df) == 1 and ((owner == 0 and dr == -2) or (owner == 1 and dr == 2)))

def can_move_as_lance(owner: int, df: int, dr: int) -> bool:
    return (df == 0 and ((owner == 0 and dr < 0) or (owner == 1 and dr > 0)))

def can_move_as_pawn(owner: int, df: int, dr: int) -> bool:
    return (df == 0 and ((owner == 0 and dr == -1) or (owner == 1 and dr == 1)))

def can_move_as_bishop(df: int, dr: int) -> bool:
    return (abs(df) == abs(dr) and df != 0 )

def can_move_as_rook(df: int, dr: int) -> bool:
    return (df == 0) ^ (dr == 0)

def can_move_as_prom_bishop(df: int, dr: int) -> bool:
    return can_move_as_bishop(df, dr) or  can_move_as_gold(0, df, dr)

def can_move_as_prom_rook(df: int, dr: int) -> bool:
    return can_move_as_rook(df, dr) or  can_move_as_silver(0, df, dr)

def bishop_attack_sqs(sq: int) -> set[int]:
    """
    sq に角があるときの利きを返す。
    """
    file, rank = sq_to_file_rank(sq)
    reachable = set()
    # 斜め
    for df, dr in ((1,1), (1,-1), (-1,1), (-1,-1)):
        f = file + df
        r = rank + dr
        while 1 <= f <= 9 and 1 <= r <= 9:
            reachable.add(file_rank_to_sq(f, r))
            f += df
            r += dr
    return reachable

def prom_bishop_attack_sqs(sq: int) -> set[int]:
    """
    sq に馬があるときの利きを返す。
    """
    file, rank = sq_to_file_rank(sq)
    reachable = set()
    # 縦横1マス
    for df, dr in ((1,0), (-1,0), (0,1), (0,-1)):
        f = file + df
        r = rank + dr
        if 1 <= f <= 9 and 1 <= r <= 9:
            reachable.add(file_rank_to_sq(f, r))
    # 斜め
    for df, dr in ((1,1), (1,-1), (-1,1), (-1,-1)):
        f = file + df
        r = rank + dr
        while 1 <= f <= 9 and 1 <= r <= 9:
            reachable.add(file_rank_to_sq(f, r))
            f += df
            r += dr
    return reachable

def pieces_reachable_by_one_move(board: cs.Board, piece: int, dst_sq: int) -> list[int]:
    """
    駒の利きだけを考えたとき、dst_sq に 1手で到達可能な piece の存在マス一覧を返す。
    """
    owner = piece_owner(piece)
    if owner is None:
        return []
    base_piece = unpromote(piece)
    candidates = {piece, base_piece}
    dst_file, dst_rank = sq_to_file_rank(dst_sq)
    reachable_sqs = []

    for sq in range(81):
        p = board.piece(sq)
        if p not in candidates:
            continue
        src_file, src_rank = sq_to_file_rank(sq)
        df = dst_file - src_file
        dr = dst_rank - src_rank
        can_reach = False
        # --- 歩 ---
        if piece in (cs.BPAWN, cs.WPAWN):
            can_reach = can_move_as_pawn(owner, df, dr)
        # --- 香 ---
        elif piece in (cs.BLANCE, cs.WLANCE):
            can_reach = can_move_as_lance(owner, df, dr)
        # --- 桂 ---
        elif piece in (cs.BKNIGHT, cs.WKNIGHT):
            can_reach = can_move_as_knight(owner, df, dr)
        # --- 銀 ---
        elif piece in (cs.BSILVER, cs.WSILVER):
            can_reach = can_move_as_silver(owner, df, dr)
        # --- 金 ---
        elif piece in (cs.BGOLD, cs.WGOLD):
            can_reach = can_move_as_gold(owner, df, dr)
        # --- 角 ---
        elif piece in (cs.BBISHOP, cs.WBISHOP):
            can_reach = can_move_as_bishop(df, dr)
        # --- 飛 ---
        elif piece in (cs.BROOK, cs.WROOK):
            can_reach = can_move_as_rook(df, dr)
        # --- と金 ---
        elif piece in (cs.BPROM_PAWN, cs.WPROM_PAWN):
            if p in (cs.BPROM_PAWN, cs.WPROM_PAWN):
                can_reach = can_move_as_gold(owner, df, dr)
            else:
                can_reach = can_move_as_pawn(owner, df, dr) and can_promote_on_move(owner, src_rank, dst_rank)
        # --- 成香 ---
        elif piece in (cs.BPROM_LANCE, cs.WPROM_LANCE):
            if p in (cs.BPROM_LANCE, cs.WPROM_LANCE):
                can_reach = can_move_as_gold(owner, df, dr)
            else:
                can_reach = can_move_as_lance(owner, df, dr) and can_promote_on_move(owner, src_rank, dst_rank)
        # --- 成桂 ---
        elif piece in (cs.BPROM_KNIGHT, cs.WPROM_KNIGHT):
            if p in (cs.BPROM_KNIGHT, cs.WPROM_KNIGHT):
                can_reach = can_move_as_gold(owner, df, dr)
            else:
                can_reach = can_move_as_knight(owner, df, dr) and can_promote_on_move(owner, src_rank, dst_rank)
        # --- 成銀 ---
        elif piece in (cs.BPROM_SILVER, cs.WPROM_SILVER):
            if p in (cs.BPROM_SILVER, cs.WPROM_SILVER):
                can_reach = can_move_as_gold(owner, df, dr)
            else:
                can_reach = can_move_as_silver(owner, df, dr) and can_promote_on_move(owner, src_rank, dst_rank)
        # --- 馬 ---
        elif piece in (cs.BPROM_BISHOP, cs.WPROM_BISHOP):
            if p in (cs.BPROM_BISHOP, cs.WPROM_BISHOP):
                can_reach = can_move_as_bishop(df, dr) or can_move_as_gold(owner, df, dr)
            else:
                can_reach = can_move_as_bishop(df, dr) and can_promote_on_move(owner, src_rank, dst_rank)
        # --- 龍 ---
        elif piece in (cs.BPROM_ROOK, cs.WPROM_ROOK):
            if p in (cs.BPROM_ROOK, cs.WPROM_ROOK):
                can_reach = can_move_as_rook(df, dr) or can_move_as_silver(owner, df, dr)
            else:
                can_reach = can_move_as_rook(df, dr) and can_promote_on_move(owner, src_rank, dst_rank)
        # --- 玉 ---
        elif piece in (cs.BKING, cs.WKING):
            can_reach = can_move_as_gold(owner, df, dr) or can_move_as_silver(owner, df, dr)
        if can_reach:
            reachable_sqs.append(sq)
    return reachable_sqs

def is_reachable_by_one_move(board: cs.Board, src_sq: int, dst_sq: int, dst_piece:int) -> bool:
    """
    src_sq にある駒が１手で dst_sq に駒種 dst_piece で到達可能かを返す。
    """
    p = board.piece(src_sq)
    if piece_owner(p) is None:
        return False
    reachable_sqs = pieces_reachable_by_one_move(board, dst_piece, dst_sq)
    return src_sq in reachable_sqs

def get_min_move_cost(piece: int, src_sq: int, dst_sq: int, need_prom: bool) -> int:
    """
    src_sq にある駒 piece が dst_sq に到達するのに掛かる最小手数を返す。
    ただし、need_prom が True の場合は途中で必ず成る。False の場合は途中で成ってはいけない。
    到達不能の場合は充分大きい値を返す。
    """
    owner = piece_owner(piece)
    if owner is None:
        return INF
    base_piece = unpromote(piece)
    # 0=生駒 1=成駒
    start_prom = 1 if piece != base_piece else 0

    q = deque()
    dist = {}
    start = (src_sq, start_prom)
    q.append(start)
    dist[start] = 0

    while q:
        sq, prom = q.popleft()
        d = dist[(sq, prom)]

        if sq == dst_sq:
            if need_prom and prom == 1:
                return d
            if not need_prom and prom == 0:
                return d

        f, r = sq_to_file_rank(sq)

        for nf in range(1, 10):
            for nr in range(1, 10):
                if nf == f and nr == r:
                    continue
                df = nf - f
                dr = nr - r
                can = False

                if prom == 1:
                    if base_piece in (cs.BPAWN, cs.WPAWN,
                                      cs.BLANCE, cs.WLANCE,
                                      cs.BKNIGHT, cs.WKNIGHT,
                                      cs.BSILVER, cs.WSILVER):
                        can = can_move_as_gold(owner, df, dr)

                    elif base_piece in (cs.BBISHOP, cs.WBISHOP):
                        can = (
                            can_move_as_bishop(df, dr)
                            or can_move_as_gold(owner, df, dr)
                        )

                    elif base_piece in (cs.BROOK, cs.WROOK):
                        can = (
                            can_move_as_rook(df, dr)
                            or can_move_as_silver(owner, df, dr)
                        )
                else:
                    if base_piece in (cs.BPAWN, cs.WPAWN):
                        can = can_move_as_pawn(owner, df, dr)

                    elif base_piece in (cs.BLANCE, cs.WLANCE):
                        can = can_move_as_lance(owner, df, dr)

                    elif base_piece in (cs.BKNIGHT, cs.WKNIGHT):
                        can = can_move_as_knight(owner, df, dr)

                    elif base_piece in (cs.BSILVER, cs.WSILVER):
                        can = can_move_as_silver(owner, df, dr)

                    elif base_piece in (cs.BGOLD, cs.WGOLD):
                        can = can_move_as_gold(owner, df, dr)

                    elif base_piece in (cs.BBISHOP, cs.WBISHOP):
                        can = can_move_as_bishop(df, dr)

                    elif base_piece in (cs.BROOK, cs.WROOK):
                        can = can_move_as_rook(df, dr)

                    elif base_piece in (cs.BKING, cs.WKING):
                        can = (
                            can_move_as_gold(owner, df, dr)
                            or can_move_as_silver(owner, df, dr)
                        )

                if not can:
                    continue

                nsq = file_rank_to_sq(nf, nr)

                # すでに成っている or 成れない
                if prom == 1 or not can_promote_on_move(owner, r, nr):
                    state = (nsq, prom)
                    if state not in dist:
                        dist[state] = d + 1
                        q.append(state)
                else:
                    # 成らない
                    s0 = (nsq, 0)
                    if s0 not in dist:
                        dist[s0] = d + 1
                        q.append(s0)
                    # 成る
                    s1 = (nsq, 1)
                    if s1 not in dist:
                        dist[s1] = d + 1
                        q.append(s1)
    return INF

def try_add_prev_board(result: list[cs.Board], prev: cs.Board, mv: int,
                       board: cs.Board = None) -> None:
    """
    １手前局面候補 prev において mv を指した場合に合法な局面なら result に prev を追加する。

    board を渡した場合は、prev で mv を指した結果が本当に board になることまで確認する。
    """
    if is_check_for_other_side(prev):
        return
    if are_kings_adjacent(prev):
        return
    if has_nifu(prev):
        return
    if not prev.is_legal(mv):
        return
    if board is not None:
        test = prev.copy()
        test.push(mv)
        if test.zobrist_hash() != board.zobrist_hash():
            return
    result.append(prev)

def previous_boards(board: cs.Board) -> list[cs.Board]:
    """
    合法な１手前の局面を返す。

    実体は retro.previous_positions に移した。旧実装には
      - 成る手の USI に "+" が付いておらず、成って移動した１手前局面を
        取り違えていた
      - 出発マスを 81 マス総当たりしていて非常に遅い
    という問題があった。
    """
    from retro import previous_positions
    return [prev for prev, _usi in previous_positions(board)]

