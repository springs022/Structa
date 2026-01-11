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
import math
import datetime
from collections import OrderedDict
from typing import List
from io_utils import (
    out
)
from validation import (
    adjust_target_turn,
    validate_piece_counts,
    is_move_touching_fixed_piece
)
from board_utils import (
    count_total_hand_num,
    get_boards_hash_from_usi,
    m_distance_vec
)
from cost_calc import (
    available_moves_for_side,
    count_position_diffs,
    need_moves_count
)

####################
# 置換表操作
####################
def tt_hit(tt: OrderedDict, h: int, remain: int, stats: dict, margin: int) -> bool:
    stats["lookups"] += 1
    failed_remain = tt.get(h)
    if failed_remain is None:
        return False
    delta = failed_remain - remain
    if delta == 0 or delta > margin:
        stats["hits"] += 1
        tt.move_to_end(h)
        return True
    return False

def tt_store(tt: OrderedDict, h: int, remain: int, max_size: int, stats: dict):
    prev = tt.get(h)
    if prev is None:
        tt[h] = remain
        tt.move_to_end(h)
        stats["stores"] += 1
    elif remain > prev:
        tt[h] = remain
        tt.move_to_end(h)
        stats["store_updates"] += 1
    else:
        return
    if len(tt) > max_size:
        tt.popitem(last=False)
        stats["evictions"] += 1

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
                             debug_usis: List[str]):

    adjust_target_turn(start_board, target_board, max_depth)
    validate_piece_counts(start_board, target_board)

    target_hash = target_board.zobrist_hash()
    solutions = []
 
    # 探索スタック (depth, iterator, found_solution)
    stack = []
    board = start_board
    path = []

    # 到達不能置換表
    TT_ENTRY_SIZE = 200
    TT_MAX_SIZE = (tt_memory_mb * 1024 * 1024) // TT_ENTRY_SIZE
    unreachable_tt = OrderedDict()

    # 統計
    total_nodes = 0
    pruned_diff_hand_s = 0
    pruned_diff_hand_g = 0
    pruned_need_moves = 0
    pruned_by_depth = [0] * (max_depth + 1)
    tt_stats = {
        "lookups": 0,
        "hits": 0,
        "stores": 0,
        "store_updates": 0,
        "evictions": 0,
    }

    # DEBUG
    if debug_usis:
        h_sols = get_boards_hash_from_usi(start_board, debug_usis)
    else:
        h_sols = []
    if h_sols:
        out(f"h_solの長さ：{len(h_sols)}", 0, True)

    # 初期状態
    stack.append((0, iter(board.legal_moves), False))

    while stack:
        depth, it, found_solution = stack[-1]
        remain = max_depth - depth

        # TT 判定
        h = board.zobrist_hash()
        if tt_hit(unreachable_tt, h, remain, tt_stats, margin):
            stack.pop()
            if path:
                board.pop()
                path.pop()
            continue

        # 終端
        if depth == max_depth:
            if h == target_hash:
                solutions.append(list(path))
                stack[-1] = (depth, it, True)
                found_solution = True
                if len(solutions) >= limit:
                    break
            else:
                tt_store(unreachable_tt, h, 0, TT_MAX_SIZE, tt_stats)
            stack.pop()
            if path:
                board.pop()
                path.pop()
            if stack:
                d, it2, f2 = stack[-1]
                stack[-1] = (d, it2, f2 or found_solution)
            continue

        # 次の手
        try:
            mv = next(it)
        except StopIteration:
            depth, it, found_solution = stack[-1]
            stack.pop()
            if not found_solution:
                tt_store(unreachable_tt, h, remain, TT_MAX_SIZE, tt_stats)
            if path:
                board.pop()
                path.pop()
            if stack:
                d, it2, f2 = stack[-1]
                stack[-1] = (d, it2, f2 or found_solution)
            continue

        # 不動駒チェック
        if is_move_touching_fixed_piece(mv, fixed_rfs):
            continue

        # 着手
        board.push(mv)
        path.append(mv)
        total_nodes += 1

        # 進捗
        if total_nodes % 200000 == 0:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            out(f"\r[{now}] {total_nodes:,} ノード探索済（検出解数：{len(solutions)}）", 1, True, False, True)

        remain_child = max_depth - (depth + 1)

        # 盤上手数計算
        avail_s = available_moves_for_side(remain_child, board.turn, 0)
        avail_g = available_moves_for_side(remain_child, board.turn, 1)
        need_s, need_g = need_moves_count(board, target_board)
        if need_s > avail_s or need_g > avail_g:
            ### DEBUG ###
            if len(h_sols) > 0:
                for i, h_sol in enumerate(h_sols):
                    if h == h_sol:
                        text = KIF.board_to_bod(board)
                        out(f"手数計算の結果、{i + 1}手目の局面が枝刈りされました。", 1)
                        out(f"need_s：{need_s}、avail_s：{avail_s}", 1)
                        out(f"need_g：{need_g}、avail_s：{avail_g}", 1)
                        out("", 1)
                        out(text, 1)
                        out("----------", 1)
            #############
            pruned_need_moves += 1
            pruned_by_depth[depth] += 1
            board.pop()
            path.pop()
            continue

        # 持駒チェック
        need_hand_s = m_distance_vec(board.pieces_in_hand[0], target_board.pieces_in_hand[0])
        need_hand_g = m_distance_vec(board.pieces_in_hand[1], target_board.pieces_in_hand[1])
        if need_hand_s > avail_s:
            pruned_diff_hand_s += 1
            pruned_by_depth[depth] += 1
            board.pop()
            path.pop()
            continue
        if need_hand_g > avail_g:
            pruned_diff_hand_g += 1
            pruned_by_depth[depth] += 1
            board.pop()
            path.pop()
            continue

        # 子ノードへ
        stack.append((depth + 1, iter(board.legal_moves), False))

    stats = {
        "total_nodes": total_nodes,
        "pruned_diff_hand_s": pruned_diff_hand_s,
        "pruned_diff_hand_g": pruned_diff_hand_g,
        "pruned_need_moves": pruned_need_moves,
        "pruned_by_depth": pruned_by_depth,
        "tt_lookups": tt_stats["lookups"],
        "tt_hits": tt_stats["hits"],
        "tt_stores": tt_stats["stores"],
        "tt_store_updates": tt_stats["store_updates"],
        "tt_evictions": tt_stats["evictions"],
        "tt_size": len(unreachable_tt),
        "tt_max_size": TT_MAX_SIZE,
    }

    return solutions, stats
