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
from typing import Optional
from typing import Tuple
from typing import List

HAND_PIECE_TO_USI = {
    cs.HPAWN:   "P",
    cs.HLANCE:  "L",
    cs.HKNIGHT: "N",
    cs.HSILVER: "S",
    cs.HGOLD:   "G",
    cs.HBISHOP: "B",
    cs.HROOK:   "R",
}

HAND_TO_PIECE = {
    0: 1,  # 歩
    1: 2,  # 香
    2: 3,  # 桂
    3: 4,  # 銀
    4: 7,  # 金
    5: 5,  # 角
    6: 6,  # 飛
}

PIECE_TO_HAND = {v: k for k, v in HAND_TO_PIECE.items()}

PROM_PIECES = {
    cs.BPROM_PAWN, cs.WPROM_PAWN,
    cs.BPROM_LANCE, cs.WPROM_LANCE,
    cs.BPROM_KNIGHT, cs.WPROM_KNIGHT,
    cs.BPROM_SILVER, cs.WPROM_SILVER,
    cs.BPROM_BISHOP, cs.WPROM_BISHOP,
    cs.BPROM_ROOK, cs.WPROM_ROOK,
}

def demote(p: int) -> int | None:
    """
    成駒 → 生駒。成駒でなければ None
    """
    return {
        cs.BPROM_PAWN:   cs.BPAWN,
        cs.BPROM_LANCE:  cs.BLANCE,
        cs.BPROM_KNIGHT: cs.BKNIGHT,
        cs.BPROM_SILVER: cs.BSILVER,
        cs.BPROM_BISHOP: cs.BBISHOP,
        cs.BPROM_ROOK:   cs.BROOK,
        cs.WPROM_PAWN:   cs.WPAWN,
        cs.WPROM_LANCE:  cs.WLANCE,
        cs.WPROM_KNIGHT: cs.WKNIGHT,
        cs.WPROM_SILVER: cs.WSILVER,
        cs.WPROM_BISHOP: cs.WBISHOP,
        cs.WPROM_ROOK:   cs.WROOK,
    }.get(p)

def piece_to_hand_piece(piece: int) -> int | None:
    """
    PIECES の piece から対応する HAND_PIECE を返す。
    王・NONE・NOTUSE などは None。
    """
    if piece in PROM_PIECES:
        piece -= 8
    if piece >= cs.WPAWN:
        piece -= (cs.WPAWN - cs.BPAWN)
    return PIECE_TO_HAND.get(piece)

def piece_value_to_name(v: int) -> str:
    table = {
        0:  "空き",
        1:  "先手歩",
        2:  "先手香",
        3:  "先手桂",
        4:  "先手銀",
        5:  "先手角",
        6:  "先手飛",
        7:  "先手金",
        8:  "先手玉",
        9:  "先手と",
        10: "先手杏",
        11: "先手圭",
        12: "先手全",
        13: "先手馬",
        14: "先手龍",
        15: "不使用",
        16: "不使用",
        17: "後手歩",
        18: "後手香",
        19: "後手桂",
        20: "後手銀",
        21: "後手角",
        22: "後手飛",
        23: "後手金",
        24: "後手玉",
        25: "後手と",
        26: "後手杏",
        27: "後手圭",
        28: "後手全",
        29: "後手馬",
        30: "後手龍",
    }
    return table.get(v, f"未知({v})")

def piece_owner(piece: int) -> Optional[int]:
    """
    cshogi の駒定数（0～30）を受け取り、0=先手(B), 1=後手(W) を返す。
    """
    # 先手
    if 1 <= piece <= 14:
        return 0
    # 後手
    if 17 <= piece <= 30:
        return 1
    # その他
    return None

def is_promoted(piece: int) -> bool:
    return (9 <= piece <= 14) or (25 <= piece <= 30)

def unpromote(piece: int) -> int:
    if is_promoted(piece):
        return piece - 8
    return piece

def change_owner(piece: int) -> Optional[int]:
    if piece in (cs.NONE, cs.NOTUSE):
        return None
    if piece <= 16:
        return piece + 16
    else:
        return piece - 16

def normalize(owner, piece, sq):
    """
    owner が後手なら、先手相当の piece と file、rank を返す。
    """
    file, rank = sq_to_file_rank(sq)
    if owner != cs.BLACK:
        file = 10 - file
        rank = 10 - rank
        piece = change_owner(piece)
    return piece, file, rank

def normalize_piece(piece: int) -> int:
    """
    PIECES の値を PIECE_TYPES_WITH_NONE に変換する。
    """
    if piece == cs.NONE:
        return cs.NONE
    if piece == cs.NOTUSE:
        return cs.NONE
    return unpromote(piece % 16)

def in_prom_zone(owner: int, rank: int) -> bool:
    if owner == 0:
        return rank <= 3
    elif owner == 1:
        return rank >= 7
    else:
        return False

def exists_prom(board: cs.Board) -> bool:
    for sq in range(81):
        if board.piece(sq) in PROM_PIECES:
            return True
    return False

def file_rank_to_sq(file: int, rank: int) -> int:
    """
    (筋, 段) → square index (0～80)
    """
    return (file - 1) * 9 + (rank - 1)

def sq_to_file_rank(sq: int) -> Tuple[int, int]:
    """
    square index (0～80) → (筋, 段)
    """
    file = sq // 9 + 1  # 1～9
    rank = sq % 9 + 1   # 1～9
    return file, rank

def file_rank_str_from_sq(sq: int) -> str:
    """
    square index (0～80) → 筋段
    """
    f, r = sq_to_file_rank(sq)
    return f"{f}{r}"

def sq_to_usi(sq: int) -> str:
    """
    square (0–80) を USI 座標（例: '7g'）に変換する。
    """
    file = sq // 9
    rank = sq % 9
    usi_file = str(1 + file)
    usi_rank = chr(ord('a') + rank)
    return usi_file + usi_rank

def c_distance(src_sq: int, dst_sq: int) -> int:
    """
    ２つの square index 間のチェビシェフ距離を返す。
    """
    f1, r1 = sq_to_file_rank(src_sq)
    f2, r2 = sq_to_file_rank(dst_sq)
    return max(abs(f1 - f2), abs(r1 - r2))

def m_distance(src_sq: int, dst_sq: int) -> int:
    """
    ２つの square index 間のマンハッタン距離を返す。
    """
    f1, r1 = sq_to_file_rank(src_sq)
    f2, r2 = sq_to_file_rank(dst_sq)
    return abs(f1 - f2) + abs(r1 - r2)

def m_distance_vec(a: List[int], b: List[int]) -> int:
    """
    同じ次元の2つの整数リスト a, b のマンハッタン距離を返す。
    """
    if len(a) != len(b):
        raise ValueError("Lists must have the same length")
    return sum(abs(x - y) for x, y in zip(a, b))

def count_pieces(board: cs.Board):
    """
    成生と先後を問わず、盤上＋持駒の各駒種の総数を数える。
    """
    counts = {}

    # 盤上の駒
    for sq in range(81):
        p = board.piece(sq)
        if p == 0:
            continue
        pt = abs(p) & 0x07  # 成生と先後を無視して駒種だけ
        counts[pt] = counts.get(pt, 0) + 1

    # 持駒（先手・後手）
    hands0, hands1 = board.pieces_in_hand

    for pt, n in enumerate(hands0):
        if n:
            pt2 = HAND_TO_PIECE[pt]
            counts[pt2] = counts.get(pt2, 0) + n
    for pt, n in enumerate(hands1):
        if n:
            pt2 = HAND_TO_PIECE[pt]
            counts[pt2] = counts.get(pt2, 0) + n
    return counts

def count_total_hand_num(board: cs.Board) -> int:
    """
    持駒の総枚数を返す。
    """
    return sum(board.pieces_in_hand[0]) + sum(board.pieces_in_hand[1])

def get_boards_hash_from_usi(start_board: cs.Board, usi_strs: List[str]) -> List[int]:
    """
    usi形式の手順を実行した局面リストを返す。
    """
    board = start_board.copy()
    boards_hash = [board.zobrist_hash()]
    for csa in usi_strs:
        board.push_usi(csa)
        boards_hash.append(board.zobrist_hash())
    return boards_hash

def hand_piece_to_board_pieces(hand_piece: int, side: int) -> set[int]:
    """
    HAND_PIECE と side から、対応する盤上駒種（不成＋成）を返す。
    side: 0 = Black, 1 = White
    """
    base_piece = HAND_TO_PIECE[hand_piece]
    if side == 1:
        base_piece += cs.WPAWN - cs.BPAWN
    result = {base_piece}
    prom_piece = base_piece + 8
    if prom_piece in PROM_PIECES:
        result.add(prom_piece)
    return result

def is_dead_end_piece(piece: int, side: int, rank: int) -> bool:
    """
    piece が rank 段目にあるとき、行き所のない駒かどうかを返す。
    side: 0 = 先手, 1 = 後手
    rank: 1～9
    """
    if side == 0:
        last = 1
        second_last = 2
    else:
        last = 9
        second_last = 8
    if piece in (cs.BPAWN, cs.BLANCE, cs.WPAWN, cs.WLANCE):
        return rank == last
    if piece in (cs.BKNIGHT, cs.WKNIGHT):
        return rank in (last, second_last)
    return False

def is_check_for_other_side(board: cs.Board) -> bool:
    """
    手番ではない方の玉に王手が掛かっているかどうかを返す。
    True なら非合法な局面。
    """
    tmp = board.copy()
    tmp.turn = 1 - tmp.turn
    return tmp.is_check()

def are_kings_adjacent(board: cs.Board) -> bool:
    """
    双方の玉が隣接しているかどうかを返す。
    """
    bk_sq = board.king_square(cs.BLACK)
    wk_sq = board.king_square(cs.WHITE)
    bf, br = sq_to_file_rank(bk_sq)
    wf, wr = sq_to_file_rank(wk_sq)
    return abs(bf - wf) <= 1 and abs(br - wr) <= 1

import cshogi as cs

import cshogi as cs

def has_nifu(board: cs.Board) -> bool:
    """
    盤面に二歩が発生しているかを返す。
    """
    for side in (0, 1):
        pawn = cs.BPAWN if side == 0 else cs.WPAWN
        for file in range(1, 10):
            found = False
            for rank in range(1, 10):
                sq = file_rank_to_sq(file, rank)
                if board.piece(sq) == pawn:
                    if found:
                        return True
                    found = True
    return False
