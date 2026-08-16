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
"""
目標局面からの逆算で作る「終端フロンティア」。

探索木のノードの大半は最終層に集中している。
旧来の探索は深さ max_depth-1 の局面から全合法手（80〜100手）を生成し、
1 手ずつ指してハッシュを目標局面と比べていた。

ここでは目標局面から k 手（1 または 2）逆算した局面の集合をあらかじめ作り、
深さ max_depth-k の局面をその集合と照合するだけで打ち切る。
最終 k 層の展開が丸ごと消えるので、k=2 なら理屈上ノード訪問が 2 桁減る。

**解の取りこぼしは起きない**（逆算は網羅的で、かつ 1 件ずつ
「その局面でその手が合法で、指すと目標局面になる」ことを確認して登録する）。
"""
import time

import cshogi as cs

from board_utils import (
    PROM_PIECES,
    HAND_PIECE_TO_USI,
    demote,
    piece_owner,
    piece_to_hand_piece,
    hand_piece_to_board_pieces,
    sq_to_usi,
    sq_to_file_rank,
    in_prom_zone,
    is_dead_end_piece,
    has_nifu,
    are_kings_adjacent,
    is_check_for_other_side,
    position_key,
)

_MASK64 = 0xFFFFFFFFFFFFFFFF

# 逆算層のエントリ数の上限。超えたら k を 1 段下げる。
DEFAULT_MAX_ENTRIES = 2_000_000

# AUTO 時だけ使う保守的な予算。短手数では逆算の準備コストが勝ちにくいため、
# 11 手以上に限り、1 手目・2 手目をこの範囲で段階的に試す。
AUTO_MIN_DEPTH = 11
AUTO_FIRST_MAX_ENTRIES = 5_000
AUTO_FIRST_MAX_SECONDS = 0.5
AUTO_SECOND_MAX_ENTRIES = 100_000
AUTO_SECOND_MAX_SECONDS = 2.0

_KING_PIECES = (cs.BKING, cs.WKING)
_PAWN_PIECES = (cs.BPAWN, cs.WPAWN)


class FrontierTooLarge(Exception):
    """逆算層が大きくなりすぎたことを表す。"""


####################
# 着手可能な出発マスの列挙
####################
# 先手視点の移動ベクトル（df = 到達筋 - 出発筋、dr = 到達段 - 出発段）。
# 段は小さいほど敵陣側なので、前進は dr = -1。
_GOLD_STEPS = ((0, -1), (-1, -1), (1, -1), (-1, 0), (1, 0), (0, 1))
_SILVER_STEPS = ((0, -1), (-1, -1), (1, -1), (-1, 1), (1, 1))
_KING_STEPS = (
    (0, -1), (-1, -1), (1, -1), (-1, 0), (1, 0), (0, 1), (-1, 1), (1, 1)
)
_BISHOP_DIRS = ((-1, -1), (1, -1), (-1, 1), (1, 1))
_ROOK_DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))

# 駒種 → (単歩の移動ベクトル, 走りの方向ベクトル)
_BLACK_MOVES = {
    cs.BPAWN:        (((0, -1),), ()),
    cs.BLANCE:       ((), ((0, -1),)),
    cs.BKNIGHT:      (((-1, -2), (1, -2)), ()),
    cs.BSILVER:      (_SILVER_STEPS, ()),
    cs.BGOLD:        (_GOLD_STEPS, ()),
    cs.BKING:        (_KING_STEPS, ()),
    cs.BBISHOP:      ((), _BISHOP_DIRS),
    cs.BROOK:        ((), _ROOK_DIRS),
    cs.BPROM_PAWN:   (_GOLD_STEPS, ()),
    cs.BPROM_LANCE:  (_GOLD_STEPS, ()),
    cs.BPROM_KNIGHT: (_GOLD_STEPS, ()),
    cs.BPROM_SILVER: (_GOLD_STEPS, ()),
    cs.BPROM_BISHOP: (_ROOK_DIRS, _BISHOP_DIRS),
    cs.BPROM_ROOK:   (_BISHOP_DIRS, _ROOK_DIRS),
}

def _build_move_table() -> dict:
    table = {}
    for piece, (steps, slides) in _BLACK_MOVES.items():
        table[piece] = (steps, slides)
        # 後手は段方向だけ反転する（筋方向に前後の区別はない）
        table[piece + 16] = (
            tuple((df, -dr) for df, dr in steps),
            tuple((df, -dr) for df, dr in slides),
        )
    return table

_MOVE_TABLE = _build_move_table()


def from_squares(piece: int, to_sq: int, pieces: list) -> list:
    """
    piece が to_sq へ 1 手で移動できる出発マスを返す。

    - 出発マスは現局面で空いていなければならない（駒はそこから離れたため）
    - 走り駒は途中に駒があればその先へ進めない
      （出発マスと到達マスの間の駒配置は、その 1 手では変化しない）

    盤の利きとしては厳密で、過不足なく列挙する。
    """
    entry = _MOVE_TABLE.get(piece)
    if entry is None:
        return []
    steps, slides = entry
    to_file, to_rank = sq_to_file_rank(to_sq)
    result = []
    for df, dr in steps:
        f = to_file - df
        r = to_rank - dr
        if 1 <= f <= 9 and 1 <= r <= 9:
            sq = (f - 1) * 9 + (r - 1)
            if pieces[sq] == cs.NONE:
                result.append(sq)
    for df, dr in slides:
        f = to_file - df
        r = to_rank - dr
        while 1 <= f <= 9 and 1 <= r <= 9:
            sq = (f - 1) * 9 + (r - 1)
            if pieces[sq] != cs.NONE:
                break
            result.append(sq)
            f -= df
            r -= dr
    return result


####################
# 1手前局面の列挙
####################
def _make_prev(board: cs.Board, pieces_prev: list, hands_prev: tuple,
               prev_turn: int, usi: str, target_hash: int,
               pawn_touched: bool, king_touched: bool):
    """
    1手前局面の候補を組み立て、成立するなら (prev_board, usi) を返す。成立しなければ None。

    「成立する」とは
      - prev が局面として非合法でない（二歩・玉の隣接・手番でない側への王手が無い）
      - usi の着手が prev で合法
      - prev でその手を指すと board になる
    のすべてを満たすこと。最後の確認を入れているので、
    ここを通ったものは必ず正しい 1 手前局面である。
    """
    prev = board.copy()
    prev.set_pieces(pieces_prev, hands_prev)
    prev.turn = prev_turn
    # 二歩・玉の隣接は、歩／玉が絡む逆算でしか新たに発生しない。
    # 元局面は合法なので、絡まないときは調べる必要がない。
    if pawn_touched and has_nifu(prev):
        return None
    if king_touched and are_kings_adjacent(prev):
        return None
    # 手番でない側に王手が掛かっている局面には到達できない
    if is_check_for_other_side(prev):
        return None
    try:
        mv = prev.move_from_usi(usi)
    except Exception:
        return None
    if not prev.is_legal(mv):
        return None
    test = prev.copy()
    test.push(mv)
    if (test.zobrist_hash() & _MASK64) != target_hash:
        return None
    return prev, usi


def previous_positions(board: cs.Board):
    """
    board の 1 手前として成立する (prev_board, usi) を列挙する。

    列挙の網羅性：
      直前に指したのは prev_turn 側なので、その駒は board 上の
      「prev_turn の駒があるマス」= 着手の到達マスに必ず残っている。
      到達マスごとに
        - 駒打ちだったか
        - 盤上移動だったか（成った／成らなかった、駒を取った／取らなかった）
      をすべて枚挙する。
    """
    prev_turn = 1 - board.turn
    cur_turn = board.turn
    pieces_org = board.pieces
    hands_org = board.pieces_in_hand
    target_hash = board.zobrist_hash() & _MASK64

    for to_sq in range(81):
        p = pieces_org[to_sq]
        if p == cs.NONE or piece_owner(p) != prev_turn:
            continue
        to_rank = to_sq % 9 + 1

        ### 駒打ち ###
        if p not in PROM_PIECES and p not in _KING_PIECES:
            hand = piece_to_hand_piece(p)
            if hand is not None:
                pieces_prev = pieces_org.copy()
                pieces_prev[to_sq] = cs.NONE
                hands_prev = (hands_org[0].copy(), hands_org[1].copy())
                hands_prev[prev_turn][hand] += 1
                got = _make_prev(
                    board, pieces_prev, hands_prev, prev_turn,
                    HAND_PIECE_TO_USI[hand] + "*" + sq_to_usi(to_sq),
                    target_hash,
                    p in _PAWN_PIECES, False,
                )
                if got is not None:
                    yield got

        ### 盤上移動 ###
        # 指す前の駒の候補。
        #   (p, False)    … 成る手ではなかった場合。指す前の駒は到達マスの駒そのもの。
        #                    p が成駒なら「すでに成っていた駒がそのまま動いた」ケースで、
        #                    出発マスも成駒の動きで引く。
        #   (base, True)  … この手で成った場合。指す前は対応する生駒。
        # 到達マスが成駒のときは両方を展開する（どちらも起こりうるため）。
        p_candidates = [(p, False)]
        if p in PROM_PIECES:
            base = demote(p)
            if base is not None:
                p_candidates.append((base, True))

        to_in_zone = in_prom_zone(prev_turn, to_rank)
        for p_prev, promoted_move in p_candidates:
            pawn_mover = p_prev in _PAWN_PIECES
            king_mover = p_prev in _KING_PIECES
            for from_sq in from_squares(p_prev, to_sq, pieces_org):
                if promoted_move:
                    from_rank = from_sq % 9 + 1
                    if not (to_in_zone or in_prom_zone(prev_turn, from_rank)):
                        continue
                usi = sq_to_usi(from_sq) + sq_to_usi(to_sq)
                if promoted_move:
                    usi += "+"

                # 駒を取らない移動
                pieces_prev = pieces_org.copy()
                pieces_prev[to_sq] = cs.NONE
                pieces_prev[from_sq] = p_prev
                hands_prev = (hands_org[0].copy(), hands_org[1].copy())
                got = _make_prev(
                    board, pieces_prev, hands_prev, prev_turn, usi,
                    target_hash, pawn_mover, king_mover,
                )
                if got is not None:
                    yield got

                # 駒を取る移動（取った駒は prev_turn の持駒に入っているはず）
                for hand_piece in cs.HAND_PIECES:
                    if hands_org[prev_turn][hand_piece] <= 0:
                        continue
                    for cp in hand_piece_to_board_pieces(hand_piece, cur_turn):
                        if is_dead_end_piece(cp, cur_turn, to_rank):
                            continue
                        pieces_prev = pieces_org.copy()
                        pieces_prev[to_sq] = cp
                        pieces_prev[from_sq] = p_prev
                        hands_prev = (
                            hands_org[0].copy(), hands_org[1].copy()
                        )
                        hands_prev[prev_turn][hand_piece] -= 1
                        got = _make_prev(
                            board, pieces_prev, hands_prev, prev_turn, usi,
                            target_hash,
                            pawn_mover or cp in _PAWN_PIECES,
                            king_mover,
                        )
                        if got is not None:
                            yield got


####################
# フロンティア
####################
class RetroFrontier:
    """
    目標局面から k 手逆算した局面の集合。

    table は zobrist ハッシュ → その局面から目標局面に至る
    USI 手順（k 手ぶんのタプル）の組。
    """

    __slots__ = ("k", "table", "layer_sizes", "build_seconds")

    def __init__(self, k: int, table: dict, layer_sizes: list,
                 build_seconds: float = 0.0):
        self.k = k
        self.table = table
        self.layer_sizes = layer_sizes
        self.build_seconds = build_seconds

    def payload(self):
        """プロセス間で受け渡すための最小限の表現。"""
        return (self.k, self.table, self.layer_sizes, self.build_seconds)

    @staticmethod
    def from_payload(payload):
        if payload is None:
            return None
        return RetroFrontier(*payload)


def build_table(target_board: cs.Board, k: int,
                max_entries: int = DEFAULT_MAX_ENTRIES,
                max_seconds: float = None):
    """
    目標局面から k 手逆算した表を作る。

    戻り値は (table, layer_sizes)。
    層が max_entries を超えたら FrontierTooLarge を投げる。
    """
    # 位置キー（SFEN の局面部分）で重複排除する。
    # ハッシュで重複排除すると、万一の衝突で手順を取り違えるため使わない。
    start_key = position_key(target_board.sfen())
    layer = {start_key: [target_board.copy(), [()]]}
    layer_sizes = []

    started = time.time()
    for step in range(k):
        last = (step == k - 1)
        nxt = {}
        for index, (_key, entry) in enumerate(layer.items()):
            if max_seconds is not None and (index & 31) == 0:
                if time.time() - started > max_seconds:
                    raise FrontierTooLarge(
                        f"逆算 {step + 1} 手目が時間上限 {max_seconds:.1f} 秒を超えました"
                    )
            board = entry[0]
            seqs = entry[1]
            for prev_board, usi in previous_positions(board):
                pkey = position_key(prev_board.sfen())
                slot = nxt.get(pkey)
                if slot is None:
                    if last:
                        head = prev_board.zobrist_hash() & _MASK64
                    else:
                        head = prev_board
                    slot = [head, []]
                    nxt[pkey] = slot
                    if len(nxt) > max_entries:
                        raise FrontierTooLarge(
                            f"逆算 {step + 1} 手目の局面数が上限 {max_entries:,} を超えました"
                        )
                dst = slot[1]
                for seq in seqs:
                    dst.append((usi,) + seq)
        layer_sizes.append(len(nxt))
        layer = nxt

    table = {}
    for _key, (h, seqs) in layer.items():
        uniq = tuple(dict.fromkeys(seqs))
        known = table.get(h)
        if known is None:
            table[h] = uniq
        else:
            # ハッシュ衝突。手順は照合時に検証されるので、束ねてよい。
            table[h] = known + uniq
    return table, layer_sizes


def build_frontier(target_board: cs.Board, max_depth: int, retro_plies: int,
                   max_entries: int = DEFAULT_MAX_ENTRIES, log=None):
    """
    設定値と手数から逆算手数 k を決めてフロンティアを作る。作らない場合は None。

    k の上限は 2。さらに「深さ max_depth-k のフレームより上に
    深さ 1 のフレームが残る」 よう max_depth-2 で抑える
    （進捗表示と再開位置が初手単位で数えられなくなるため）。
    """
    auto = isinstance(retro_plies, str) and retro_plies.upper() == "AUTO"
    if auto:
        if max_depth < AUTO_MIN_DEPTH:
            return None
        # まず1手だけを軽い予算で作る。これが重ければ逆算は使わない。
        try:
            t0 = time.time()
            table1, sizes1 = build_table(
                target_board, 1, min(max_entries, AUTO_FIRST_MAX_ENTRIES),
                AUTO_FIRST_MAX_SECONDS,
            )
            elapsed1 = time.time() - t0
        except FrontierTooLarge:
            return None
        if elapsed1 > AUTO_FIRST_MAX_SECONDS:
            return None

        # 2手目だけが重いときは、完成済みの1手フロンティアを使う。
        try:
            t0 = time.time()
            table2, sizes2 = build_table(
                target_board, 2, min(max_entries, AUTO_SECOND_MAX_ENTRIES),
                AUTO_SECOND_MAX_SECONDS,
            )
            elapsed2 = time.time() - t0
        except FrontierTooLarge:
            return RetroFrontier(1, table1, sizes1, elapsed1)
        if elapsed2 > AUTO_SECOND_MAX_SECONDS:
            return RetroFrontier(1, table1, sizes1, elapsed1)
        return RetroFrontier(2, table2, sizes2, elapsed1 + elapsed2)

    k = min(retro_plies, 2, max_depth - 2)
    if k <= 0:
        return None
    while k > 0:
        t0 = time.time()
        try:
            table, layer_sizes = build_table(target_board, k, max_entries)
        except FrontierTooLarge as e:
            if log:
                log(f"逆算 {k} 手：{e} → {k - 1} 手に切り下げます")
            k -= 1
            continue
        elapsed = time.time() - t0
        return RetroFrontier(k, table, layer_sizes, elapsed)
    return None


def resolve_sequence(board: cs.Board, usi_seq, target_key: str,
                     fixed_sqs=None, touches_fixed=None):
    """
    board から usi_seq の手順を実際に指してみて、目標局面に到達するなら
    その着手（cshogi の move 値）のリストを返す。到達しなければ None。

    フロンティアは登録時に検証済みだが、ここでも確認する。
      - ハッシュ衝突で無関係な手順が引かれる可能性を潰すため
      - 不動駒に触れる手順を除外するため（探索本体の着手フィルタは
        フロンティアで飛ばした手には掛からない）
    着手は board 自身の合法手生成から取り出すので、
    KIF 出力に必要な駒種情報も従来どおり正しく入る。
    """
    tmp = board.copy()
    moves = []
    for usi in usi_seq:
        found = None
        for mv in tmp.legal_moves:
            if cs.move_to_usi(mv) == usi:
                found = mv
                break
        if found is None:
            return None
        if fixed_sqs and touches_fixed is not None:
            if touches_fixed(found, fixed_sqs):
                return None
        tmp.push(found)
        moves.append(found)
    if position_key(tmp.sfen()) != target_key:
        return None
    return moves
