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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
"""一対一割当を使って、開始局面から動かせない駒を安全に証明する。"""

from collections import defaultdict

import cshogi as cs

from board_utils import (
    normalize_piece,
    piece_owner,
    piece_value_to_name,
    sq_to_file_rank,
)
from cost_calc import (
    TargetInfo,
    available_moves_for_side,
    build_target_info,
    major_p_cost,
    minor_p_cost,
    unprom_move_cost,
)


# 到達不能辺。Structa の最大手数では片側 63 手までなので、十分大きい。
_ASSIGNMENT_INF = 1_000_000


def _minimum_rect_assignment(costs: list[list[int]]) -> int:
    """行を相異なる列へ割り当てる最小費用（行数 <= 列数）。"""
    if not costs:
        return 0
    n = len(costs)
    m = len(costs[0])
    if n > m or any(len(row) != m for row in costs):
        raise ValueError("割当行列の形が不正です")

    # Hungarian algorithm。u, v は双対ポテンシャル、p は列側の対応。
    u = [0] * (n + 1)
    v = [0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        minv = [_ASSIGNMENT_INF] * (m + 1)
        used = [False] * (m + 1)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = _ASSIGNMENT_INF
            j1 = 0
            row = costs[i0 - 1]
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = row[j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return -v[0]


def _source_cost(req, src_piece: int, src_sq: int) -> int:
    """盤上の1枚を目標要求へ運ぶ、障害物を無視した安全な下界。"""
    if src_piece not in req.candidates:
        return _ASSIGNMENT_INF
    if req.promoted:
        cost_fn = major_p_cost if req.is_major else minor_p_cost
        cost = cost_fn(src_piece, src_sq, req.sq)
    else:
        cost = unprom_move_cost(src_piece, src_sq, req.sq)
    if cost is None or cost >= 100:
        return _ASSIGNMENT_INF
    return cost


def _assignment_cost_for_side(
    start_board: cs.Board,
    target_info: TargetInfo,
    side: int,
    *,
    force_stay_sq: int | None = None,
    forbid_stay_sq: int | None = None,
) -> int:
    """
    同じ所有者・基本駒種の中で、現存駒を一対一に目標駒へ割り当てる。

    各目標には make_cost の仮想生成元も用意する。捕獲による入手を数えない
    緩和なので、返り値は実際の必要手数を超えない。
    """
    sources_by_kind = defaultdict(list)
    for sq in range(81):
        piece = start_board.piece(sq)
        if piece_owner(piece) == side:
            sources_by_kind[normalize_piece(piece)].append((sq, piece))

    reqs_by_kind = defaultdict(list)
    for req in target_info.requirements:
        if req.owner == side:
            reqs_by_kind[normalize_piece(req.piece)].append(req)

    total = 0
    for kind, reqs in reqs_by_kind.items():
        sources = sources_by_kind.get(kind, ())
        source_count = len(sources)
        req_count = len(reqs)
        costs = []
        for req in reqs:
            row = [_source_cost(req, piece, sq) for sq, piece in sources]
            # 仮想生成元は必要枚数だけ置く。どの仮想列を使っても費用は同じ。
            make_cost = req.make_cost
            if make_cost >= 100:
                make_cost = _ASSIGNMENT_INF
            row.extend([make_cost] * req_count)

            if req.sq == force_stay_sq:
                for col, (src_sq, _piece) in enumerate(sources):
                    if src_sq != force_stay_sq:
                        row[col] = _ASSIGNMENT_INF
                for col in range(source_count, len(row)):
                    row[col] = _ASSIGNMENT_INF
            elif force_stay_sq is not None:
                # force_stay_sq の現存駒は同じ目標マスに使うため、他では使えない。
                for col, (src_sq, _piece) in enumerate(sources):
                    if src_sq == force_stay_sq:
                        row[col] = _ASSIGNMENT_INF

            if req.sq == forbid_stay_sq:
                for col, (src_sq, _piece) in enumerate(sources):
                    if src_sq == forbid_stay_sq:
                        row[col] = _ASSIGNMENT_INF
            costs.append(row)
        total += _minimum_rect_assignment(costs)
    return total


def prove_auto_fixed_sqs(
    start_board: cs.Board,
    target_board: cs.Board,
    max_depth: int,
    already_fixed_sqs: set[int] | None = None,
) -> set[int]:
    """
    指定手数内のどの解でも触れられない開始駒のマスを返す。

    開始・目標で同じ駒があるマス q に触れる解は、必ず次のどちらかになる。
      1. q の現存駒が最終的に q を担当する（動いて戻るので、その側に最低2手追加）
      2. 別の現存駒または生成駒が q を担当する（q→q の割当を禁止）
    両ケースの一対一割当下界がその側の手数予算を超える場合だけ証明する。
    """
    target_info = build_target_info(target_board)
    fixed = already_fixed_sqs or set()
    budgets = (
        available_moves_for_side(max_depth, start_board.turn, cs.BLACK),
        available_moves_for_side(max_depth, start_board.turn, cs.WHITE),
    )
    result = set()
    for sq in range(81):
        if sq in fixed:
            continue
        piece = start_board.piece(sq)
        if piece == cs.NONE or target_board.piece(sq) != piece:
            continue
        side = piece_owner(piece)
        if side not in (cs.BLACK, cs.WHITE):
            continue

        stay_cost = _assignment_cost_for_side(
            start_board, target_info, side, force_stay_sq=sq
        ) + 2
        replacement_cost = _assignment_cost_for_side(
            start_board, target_info, side, forbid_stay_sq=sq
        )
        if min(stay_cost, replacement_cost) > budgets[side]:
            result.add(sq)
    return result


def sqs_to_rfs(sqs: set[int]) -> set[int]:
    """cshogi の square index 集合を「筋段」の2桁整数集合へ変換する。"""
    result = set()
    for sq in sqs:
        file, rank = sq_to_file_rank(sq)
        result.add(file * 10 + rank)
    return result


def format_auto_fixed_pieces(board: cs.Board, sqs: set[int]) -> str:
    """自動不動駒を「後手51玉、先手59玉」の形式で返す。"""
    items = []
    for sq in sorted(sqs, key=lambda x: next(iter(sqs_to_rfs({x})))):
        file, rank = sq_to_file_rank(sq)
        rf = file * 10 + rank
        name = piece_value_to_name(board.piece(sq))
        items.append(f"{name[:2]}{rf}{name[2:]}")
    return "、".join(items)
