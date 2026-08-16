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
from cshogi import KIF
import random
import datetime
from typing import List
from io_utils import (
    out
)
from validation import (
    adjust_target_turn,
    validate_piece_counts,
    is_move_touching_fixed_sqs,
    rfs_to_sqs
)
from board_utils import (
    get_boards_hash_from_usi,
    hand_distance,
    position_key
)
from cost_calc import (
    available_moves_for_side,
    build_target_info,
    corrected_need_moves_count
)
from tt import (
    UnreachableTT,
    CostTT,
    TT_ENTRY_SIZE,
    COST_TT_ENTRY_SIZE
)
from retro import (
    build_frontier,
    resolve_sequence
)

_MASK64 = 0xFFFFFFFFFFFFFFFF

# コスト計算置換表のキーに深さを混ぜるための固定乱数表。
# 同一局面でも残り手数が違えば別の部分問題なので、区別する必要がある。
_DEPTH_KEY_RNG = random.Random(0x5EED5EED)
_DEPTH_KEYS = [_DEPTH_KEY_RNG.getrandbits(64) for _ in range(256)]


def _depth_keys(n: int) -> list:
    """深さ n まで使える固定乱数表を返す。"""
    while len(_DEPTH_KEYS) <= n:
        _DEPTH_KEYS.append(_DEPTH_KEY_RNG.getrandbits(64))
    return _DEPTH_KEYS


_position_key = position_key


def _new_tt_stats() -> dict:
    return {
        "lookups": 0,
        "hits": 0,
        "stores": 0,
        "store_updates": 0,
        "evictions": 0,
    }


####################
# 探索部
####################
def find_all_paths_to_target(start_board: cs.Board,
                             target_board: cs.Board,
                             max_depth: int,
                             limit: int,
                             fixed_rfs: set,
                             tt_memory_mb: int,
                             margin: int,
                             first_move_index: int,
                             previous_solutions: List[List[int]],
                             debug_usis: List[str],
                             progress_prefix: str = "",
                             retro_plies: int = 2,
                             retro=None):

    # 置換表は残り手数を 1 バイトに詰めて持つ
    if max_depth > 126:
        raise ValueError("MAX_DEPTH は 126 以下である必要があります。")

    adjust_target_turn(start_board, target_board, max_depth)
    validate_piece_counts(start_board, target_board)

    target_hash = target_board.zobrist_hash() & _MASK64
    target_key = _position_key(target_board.sfen())
    depth_keys = _depth_keys(max_depth)
    target_info = build_target_info(target_board)
    target_hands = target_board.pieces_in_hand
    target_hand_s = target_hands[0]
    target_hand_g = target_hands[1]
    solutions = list(previous_solutions)
    interrupted = False

    # 不動駒。着手フィルタ用に square index 集合を作る。
    fixed_sqs = rfs_to_sqs(fixed_rfs)
    use_fixed = bool(fixed_sqs)

    # ---- 終端フロンティア（目標局面から k 手逆算した局面集合）----
    # 深さ max_depth-k の局面をこの集合と照合するだけで打ち切れるので、
    # 最終 k 層の合法手生成と着手が丸ごと不要になる。
    if retro is None and retro_plies > 0:
        retro = build_frontier(
            target_board, max_depth, retro_plies,
            log=lambda m: out(m, 2, console=True),
        )
    retro_k = retro.k if retro is not None else 0
    if retro_k and (max_depth - retro_k) < 1:
        # 終端の深さが根より浅くなる組み合わせ（想定外）。安全側に倒して無効化する。
        retro = None
        retro_k = 0
    if retro_k:
        frontier_get = retro.table.get
        terminal_depth = max_depth - retro_k
        frontier_size = retro.layer_sizes[-1]
        out(
            f"終端フロンティア：{retro_k}手逆算"
            f"（局面数 {frontier_size:,}、構築 {retro.build_seconds:.1f}秒）",
            2, console=True
        )
    else:
        frontier_get = None
        terminal_depth = max_depth

    # 深さごとの残り手数テーブル。
    # 深さ d（= 開始局面から d 手進んだ状態）での手番は start_turn と d の偶奇で決まるため、
    # 毎ノードで available_moves_for_side を呼ぶ必要はない。
    start_turn = start_board.turn
    avail_s_at = [0] * (max_depth + 1)
    avail_g_at = [0] * (max_depth + 1)
    for d in range(max_depth + 1):
        turn_at_d = start_turn if (d % 2 == 0) else (1 - start_turn)
        avail_s_at[d] = available_moves_for_side(max_depth - d, turn_at_d, 0)
        avail_g_at[d] = available_moves_for_side(max_depth - d, turn_at_d, 1)

    # 探索スタック。フレームは [depth, iterator, found_solution, zobrist_hash]。
    # ハッシュをフレームに持たせることで、子から戻るたびの再計算をなくす。
    stack = []
    board = start_board
    path = []
    pushed = 0     # board に積んだ手数（中断時の巻き戻し用）

    # 到達不能置換表・コスト計算置換表
    COST_TT_RATIO = 0.4
    TOTAL_TT_BYTES = tt_memory_mb * 1024 * 1024
    UNREACHABLE_TT_BYTES = int(TOTAL_TT_BYTES * (1.0 - COST_TT_RATIO))
    COST_TT_BYTES = TOTAL_TT_BYTES - UNREACHABLE_TT_BYTES
    TT_MAX_SIZE = max(1024, UNREACHABLE_TT_BYTES // TT_ENTRY_SIZE)
    COST_TT_MAX_SIZE = max(1024, COST_TT_BYTES // COST_TT_ENTRY_SIZE)
    tt_stats = _new_tt_stats()
    cost_tt_stats = {"lookups": 0, "hits": 0}
    unreachable_tt = UnreachableTT(TT_MAX_SIZE, tt_stats)
    cost_tt = CostTT(COST_TT_MAX_SIZE, cost_tt_stats)
    tt_hit = unreachable_tt.hit
    tt_store = unreachable_tt.store
    cost_get = cost_tt.get
    cost_put = cost_tt.put

    # 統計
    total_nodes = 0
    pruned_diff_hand_s = 0
    pruned_diff_hand_g = 0
    pruned_need_moves = 0
    frontier_misses = 0
    pruned_by_depth = [0] * (max_depth + 1)

    # DEBUG
    if debug_usis:
        h_sols = [
            x & _MASK64 for x in get_boards_hash_from_usi(start_board, debug_usis)
        ]
    else:
        h_sols = []
    if h_sols:
        out(f"h_solの長さ：{len(h_sols)}", 0, True)

    # 初期状態
    first_moves_all = sorted(
        list(board.legal_moves),
        key=lambda mv: cs.move_to_usi(mv)
    )
    total_first_moves = len(first_moves_all)
    first_moves = first_moves_all[first_move_index:]
    stack.append([0, iter(first_moves), False, board.zobrist_hash() & _MASK64])

    def show_progress():
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if total_first_moves > 0:
            percent = int(first_move_index / total_first_moves * 100)
            out(
                f"\r[{now}] {progress_prefix}{percent}% 探索済"
                f"（検出解数：{len(solutions)}）",
                1, True, False, True
            )

    try:
        # 初回進捗表示
        show_progress()

        while stack:
            frame = stack[-1]
            depth = frame[0]

            # 次の手
            try:
                mv = next(frame[1])
            except StopIteration:
                stack.pop()
                found_solution = frame[2]
                if depth == 1:
                    first_move_index += 1
                if not found_solution and depth > 0:
                    # 部分木を掘り尽くして解が無かったので、到達不能として登録する。
                    tt_store(frame[3], max_depth - depth)
                if path:
                    board.pop()
                    path.pop()
                    pushed -= 1
                if stack and found_solution:
                    stack[-1][2] = True
                continue

            # 不動駒チェック
            if use_fixed and is_move_touching_fixed_sqs(mv, fixed_sqs):
                # 初手を弾いたときも進捗を進める。
                # 旧版はここで first_move_index を進めていなかったため、
                # 不動駒設定時に進捗率が実際より小さく表示され、
                # 再開時に検討済みの初手をやり直していた。
                if depth == 0:
                    first_move_index += 1
                continue

            # 着手
            board.push(mv)
            pushed += 1
            total_nodes += 1
            child_depth = depth + 1
            h = board.zobrist_hash() & _MASK64

            # 進捗
            if total_nodes % 100000 == 0:
                show_progress()

            # ---- 終端 ----
            # 逆算フロンティアを使う場合は深さ max_depth-k、
            # 使わない場合は深さ max_depth が終端。
            # いずれもハッシュ 1 回の照合で決着するので、
            # 合法手生成も持駒判定も盤上手数計算も要らない。
            # 末端の失敗を置換表に登録しないのも同じ理由
            # （再訪コストが小さい一方で件数が膨大なため、
            #   有用な深いエントリを追い出してしまう）。
            if child_depth == terminal_depth:
                if retro_k:
                    seqs = frontier_get(h)
                    if seqs is not None:
                        head = path + [mv]
                        for seq in seqs:
                            tail = resolve_sequence(
                                board, seq, target_key,
                                fixed_sqs if use_fixed else None,
                                is_move_touching_fixed_sqs,
                            )
                            if tail is None:
                                continue
                            new_solution = head + tail
                            if new_solution not in solutions:
                                solutions.append(new_solution)
                            frame[2] = True
                        board.pop()
                        pushed -= 1
                        if len(solutions) >= limit:
                            break
                        continue
                    frontier_misses += 1
                    board.pop()
                    pushed -= 1
                    continue
                # ハッシュ衝突で偽の解を出さないよう、一致時だけ実局面で確認する。
                if h == target_hash and _position_key(board.sfen()) == target_key:
                    new_solution = path + [mv]
                    if new_solution not in solutions:
                        solutions.append(new_solution)
                    frame[2] = True
                    board.pop()
                    pushed -= 1
                    if len(solutions) >= limit:
                        break
                    continue
                board.pop()
                pushed -= 1
                continue

            # ---- 置換表 ----
            remain_child = max_depth - child_depth
            if tt_hit(h, remain_child, margin):
                board.pop()
                pushed -= 1
                continue

            avail_s = avail_s_at[child_depth]
            avail_g = avail_g_at[child_depth]

            # 持駒チェック（盤上手数計算より軽いため先に判定）
            hands = board.pieces_in_hand
            if hand_distance(hands[0], target_hand_s) > avail_s:
                pruned_diff_hand_s += 1
                pruned_by_depth[depth] += 1
                board.pop()
                pushed -= 1
                continue
            if hand_distance(hands[1], target_hand_g) > avail_g:
                pruned_diff_hand_g += 1
                pruned_by_depth[depth] += 1
                board.pop()
                pushed -= 1
                continue

            # ---- 盤上手数計算 ----
            ck = h ^ depth_keys[child_depth]
            cached = cost_get(ck)
            if cached is None:
                need_s, need_g = corrected_need_moves_count(
                    board, target_board, avail_s, avail_g,
                    fixed_sqs, target_info, hands
                )
                cost_put(ck, need_s, need_g)
            else:
                need_s, need_g = cached
            if need_s > avail_s or need_g > avail_g:
                ### DEBUG ###
                if h_sols:
                    for i, h_sol in enumerate(h_sols):
                        if h == h_sol:
                            text = KIF.board_to_bod(board)
                            out(f"手数計算の結果、{i + 1}手目の局面が枝刈りされました。", 1)
                            out(f"need_s：{need_s}、avail_s：{avail_s}", 1)
                            out(f"need_g：{need_g}、avail_g：{avail_g}", 1)
                            out("", 1)
                            out(text, 1)
                            out("----------", 1)
                #############
                pruned_need_moves += 1
                pruned_by_depth[depth] += 1
                board.pop()
                pushed -= 1
                continue

            # 子ノードへ
            path.append(mv)
            stack.append([child_depth, iter(board.legal_moves), False, h])

        # 最終進捗表示
        show_progress()
    except KeyboardInterrupt:
        interrupted = True

    # 開始局面まで巻き戻す（旧版は解数上限での break 時に積んだままだった）
    while pushed > 0:
        board.pop()
        pushed -= 1

    stats = {
        "total_nodes": total_nodes,
        "pruned_diff_hand_s": pruned_diff_hand_s,
        "pruned_diff_hand_g": pruned_diff_hand_g,
        "pruned_need_moves": pruned_need_moves,
        "frontier_misses": frontier_misses,
        "retro_k": retro_k,
        "retro_layer_sizes": list(retro.layer_sizes) if retro is not None else [],
        "pruned_by_depth": pruned_by_depth,
        "tt_lookups": tt_stats["lookups"],
        "tt_hits": tt_stats["hits"],
        "tt_stores": tt_stats["stores"],
        "tt_store_updates": tt_stats["store_updates"],
        "tt_evictions": tt_stats["evictions"],
        "tt_size": len(unreachable_tt),
        "tt_max_size": unreachable_tt.size,
        "cost_tt_lookups": cost_tt_stats["lookups"],
        "cost_tt_hits": cost_tt_stats["hits"],
        "cost_tt_size": len(cost_tt),
        "cost_tt_max_size": cost_tt.size,
    }

    return solutions, stats, first_move_index, interrupted
