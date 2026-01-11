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
import psutil
from typing import List
import unicodedata as uni
import config 

def load_kv_file(path: str) -> dict:
    """
    key = value 形式のファイルの内容を取得する。
    """
    data = {}
    with open(path, encoding="shift_jis") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    return data

def out_file(msg=""):
    """
    ファイル出力
    """
    config.out_fp.write(msg + "\n")
    config.out_fp.flush()

def out(msg="", level=1, console=False, file=True, overwrite=False):
    """
    ログ出力用ユーティリティ。
        level : 出力レベル（output_level 以下なら出力）
        console : 標準出力に出すか
        file : ログファイルに出すか
        overwrite : 進捗表示用（改行しない）
    """
    if config.output_level >= level:
        if console:
            if overwrite:
                print(msg, end="", flush=True)
            else:
                print(msg)
        if file:
            out_file(msg)

def log_system_info():
    """
    システム情報の出力
    """
    vm = psutil.virtual_memory()
    out("実行環境情報", 3)
    out(f"CPU論理コア数：{psutil.cpu_count(logical=True)}", 3)
    out(f"CPU使用率(初期)：{psutil.cpu_percent(interval=0.1)}%", 3)
    out(f"総メモリ：{vm.total // (1024**2):,} MB", 3)
    out(f"利用可能メモリ：{vm.available // (1024**2):,} MB", 3)
    out('--------------------', 3)

def print_solution_kif(start_board: cs.Board, moves: List[int]) -> None:
    """
    KIF手順の出力
    """
    board = start_board.copy()
    prevmv = None
    for i, mv in enumerate(moves, 1):
        kif = KIF.move_to_kif(mv, prevmv)
        out(f"{i:>3} {kif}", 0)
        board.push(mv)
        prevmv = mv

def get_width_count(text):
    """
    文字列の横幅取得関数
    """
    count = 0
    for c in text:
        if uni.east_asian_width(c) in 'FWA':
            count += 2
        else:
            count += 1
    return count

def pad_to(s: str, width: int) -> str:
    """
    パディング関数
    """
    if get_width_count(s) < width:
        return s + " " * (width - get_width_count(s))
    return s

def get_boards_side_by_side(board_left: cs.Board, board_right: cs.Board, sep="        ") -> list[str]:
    """
    ２局面を横並びにした文字列の配列を返す。
    """
    left_lines  = KIF.board_to_bod(board_left).splitlines()
    right_lines = KIF.board_to_bod(board_right).splitlines()
    left_lines  = left_lines[:14]
    right_lines = right_lines[:14]
    result = []
    left_width = max(get_width_count(left_lines[i]) for i in range(0, 13))
    for i in range(14):
        line = pad_to(left_lines[i], left_width) + sep + right_lines[i]
        result.append(line)
    return result

def load_debug_sol(path: str) -> list[str]:
    """
    デバッグ用関数。ファイル内の手順を読み込んでUSI形式の指し手リストを返す。
    """
    lines = []
    in_debug = False
    with open(path, encoding="shift_jis") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "DEBUG_SOL_START":
                in_debug = True
                continue
            if line == "DEBUG_SOL_END":
                break
            if in_debug:
                lines.append(line)
    return lines
