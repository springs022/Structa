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
初手単位のマルチプロセス並列探索。

探索は初手ごとに完全に独立しているため（進捗表示も初手単位）、
初手を各ワーカープロセスに配るだけでコア数ぶんスケールする。
置換表はワーカーごとに独立に持つ（共有しない）。

再開用ファイルの意味づけは従来どおり「先頭から連続して完了した初手数」。
並列では完了順が前後するので、完了した添字集合から
「連続する先頭部分の長さ」を求めて記録する。
やり直しが発生しても正しさは損なわれない。
"""
import multiprocessing
import os
import signal
import datetime

import cshogi as cs

import config
from io_utils import out
from search import find_all_paths_to_target
from validation import (
    adjust_target_turn,
    validate_piece_counts,
    is_move_touching_fixed_sqs,
    rfs_to_sqs,
)

# ワーカープロセスごとの問題設定
_W = {}

# 親プロセスが Ctrl+C を処理するために、結果待ちを無期限にしない。
# ワーカーの探索はこの間も連続して実行される。
RESULT_WAIT_TIMEOUT_SECONDS = 0.2


def decide_process_count(requested: int) -> int:
    """PROCESSES 設定値から実際のプロセス数を決める（0 または負なら自動）。"""
    if requested and requested > 0:
        return requested
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
    except Exception:
        n = None
    if not n:
        n = os.cpu_count() or 1
    return max(1, n)


def enumerate_first_moves(start_board: cs.Board, fixed_rfs: set) -> list:
    """
    探索対象となる初手を、直列版と同じ順序（USI 文字列の昇順）で返す。

    戻り値は ((添字, USI) のリスト, 除外前の初手総数)。
    添字は不動駒で除外した手も含めた通し番号で、
    再開用ファイルの completed_first_moves と対応する。
    """
    fixed_sqs = rfs_to_sqs(fixed_rfs)
    all_moves = sorted(list(start_board.legal_moves), key=cs.move_to_usi)
    result = []
    for i, mv in enumerate(all_moves):
        if fixed_sqs and is_move_touching_fixed_sqs(mv, fixed_sqs):
            continue
        result.append((i, cs.move_to_usi(mv)))
    return result, len(all_moves)


def _worker_init(start_sfen, target_sfen, max_depth, limit,
                 fixed_rfs, tt_memory_mb, margin):
    # Ctrl+C は親プロセスだけが受け取る
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # 子プロセスからはログを一切出さない（out_fp を持っていない）
    config.output_level = -1
    _W["start_sfen"] = start_sfen
    _W["target_sfen"] = target_sfen
    _W["max_depth"] = max_depth
    _W["limit"] = limit
    _W["fixed_rfs"] = fixed_rfs
    _W["tt_memory_mb"] = tt_memory_mb
    _W["margin"] = margin


def _worker_task(item):
    """初手 1 つぶんの部分木を探索する。"""
    index, mv_usi = item
    start = cs.Board(_W["start_sfen"])
    target = cs.Board(_W["target_sfen"])
    mv = start.move_from_usi(mv_usi)
    start.push(mv)
    sols, stats, _idx, _interrupted = find_all_paths_to_target(
        start,
        target,
        _W["max_depth"] - 1,
        _W["limit"],
        _W["fixed_rfs"],
        _W["tt_memory_mb"],
        _W["margin"],
        0,
        [],
        [],
    )
    sols_usi = [
        [mv_usi] + [cs.move_to_usi(m) for m in sol]
        for sol in sols
    ]
    return index, sols_usi, stats


def _merge_stats(total: dict, part: dict, max_depth: int) -> None:
    """ワーカーの統計を親側に足し込む。手数別は 1 手ぶんずらす。"""
    for key in (
        "total_nodes", "pruned_diff_hand_s", "pruned_diff_hand_g",
        "pruned_need_moves", "tt_lookups", "tt_hits", "tt_stores",
        "tt_store_updates", "tt_evictions", "cost_tt_lookups", "cost_tt_hits",
    ):
        total[key] = total.get(key, 0) + part.get(key, 0)
    # 子は初手を指した局面を深さ 0 として数えるので +1 してから足す
    child_by_depth = part.get("pruned_by_depth", [])
    by_depth = total["pruned_by_depth"]
    for d, c in enumerate(child_by_depth):
        if d + 1 < len(by_depth):
            by_depth[d + 1] += c
    total["tt_size"] = max(total.get("tt_size", 0), part.get("tt_size", 0))
    total["tt_max_size"] = part.get("tt_max_size", 0)
    total["cost_tt_size"] = max(
        total.get("cost_tt_size", 0), part.get("cost_tt_size", 0)
    )
    total["cost_tt_max_size"] = part.get("cost_tt_max_size", 0)


def _completed_prefix(done: set, all_indices: list, total_first_moves: int) -> int:
    """
    完了した初手の添字集合から「先頭から連続して完了した数」を返す。

    不動駒で除外された初手は最初から完了扱いにする。
    """
    searched = set(all_indices)
    n = 0
    for i in range(total_first_moves):
        if i in done or i not in searched:
            n += 1
        else:
            break
    return n


def find_all_paths_to_target_parallel(start_board: cs.Board,
                                      target_board: cs.Board,
                                      max_depth: int,
                                      limit: int,
                                      fixed_rfs: set,
                                      tt_memory_mb: int,
                                      margin: int,
                                      first_move_index: int,
                                      previous_solutions: list,
                                      processes: int):
    """
    直列版 find_all_paths_to_target と同じ戻り値
    (solutions, stats, completed_first_moves, interrupted) を返す。
    """
    # 入力の妥当性は子プロセスを起こす前に親で確認しておく
    adjust_target_turn(start_board, target_board, max_depth)
    validate_piece_counts(start_board, target_board)

    start_sfen = start_board.sfen()
    target_sfen = target_board.sfen()

    pairs, total_first_moves = enumerate_first_moves(start_board, fixed_rfs)
    todo = [(i, u) for (i, u) in pairs if i >= first_move_index]

    solutions = list(previous_solutions)
    stats = {
        "total_nodes": 0,
        "pruned_diff_hand_s": 0,
        "pruned_diff_hand_g": 0,
        "pruned_need_moves": 0,
        "pruned_by_depth": [0] * (max_depth + 1),
    }
    done = set(i for (i, _u) in pairs if i < first_move_index)
    interrupted = False

    # ワーカーごとに置換表を持つので、上限メモリは分割する
    per_worker_mb = max(16, tt_memory_mb // max(1, processes))

    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(
        processes=processes,
        initializer=_worker_init,
        initargs=(start_sfen, target_sfen, max_depth, limit,
                  fixed_rfs, per_worker_mb, margin),
    )

    all_indices = [i for (i, _u) in pairs]
    found_count = [len(previous_solutions)]

    def show_progress():
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if total_first_moves > 0:
            n = _completed_prefix(done, all_indices, total_first_moves)
            percent = int(n / total_first_moves * 100)
            out(
                f"\r[{now}] {percent}% 探索済（検出解数：{found_count[0]}）",
                1, True, False, True
            )

    collected = []
    try:
        show_progress()
        result_iterator = pool.imap_unordered(_worker_task, todo, chunksize=1)
        while True:
            try:
                index, sols_usi, part_stats = result_iterator.next(
                    RESULT_WAIT_TIMEOUT_SECONDS
                )
            except multiprocessing.TimeoutError:
                # Windows では無期限の結果待ち中に Ctrl+C の処理が遅れる。
                # 定期的に Python の実行へ戻し、割り込みを受け取れるようにする。
                continue
            except StopIteration:
                break
            done.add(index)
            stats["total_nodes"] += 1   # 初手そのもののノード
            _merge_stats(stats, part_stats, max_depth)
            if sols_usi:
                collected.append((index, sols_usi))
                found_count[0] += len(sols_usi)
            show_progress()
            if found_count[0] >= limit:
                break
        show_progress()
    except KeyboardInterrupt:
        interrupted = True
    finally:
        pool.terminate()
        pool.join()

    # 直列版と同じ順（初手の USI 昇順）になるよう添字で整列する
    collected.sort(key=lambda x: x[0])
    for _index, sols_usi in collected:
        for sol_usi in sols_usi:
            board = start_board.copy()
            moves = []
            for usi in sol_usi:
                mv = board.move_from_usi(usi)
                moves.append(mv)
                board.push(mv)
            if moves not in solutions:
                solutions.append(moves)
            if len(solutions) >= limit:
                break
        if len(solutions) >= limit:
            break

    completed = _completed_prefix(done, all_indices, total_first_moves)
    return solutions, stats, completed, interrupted
