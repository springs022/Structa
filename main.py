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
import sys
import os
import time
import datetime
import argparse
import faulthandler
faulthandler.enable()
import config
from board_utils import (
    piece_value_to_name,
    file_rank_to_sq,
)
from io_utils import (
    load_kv_file,
    out,
    log_system_info,
    print_solution_kif,
    get_boards_side_by_side,
    load_debug_sol,
    save_resume_file
)
from validation import (
    validate_sfen_has_king,
    validate_two_digits,
)
from search import find_all_paths_to_target

if __name__ == "__main__":
    try:
        # 引数パース
        parser = argparse.ArgumentParser(description="Structa - Shogi Proof Game Proofer")
        parser.add_argument( 
            "-i", "--input",
            help="入力ファイル名（省略時は config.txt の INPUT_FILE を使用）"
        )
        parser.add_argument(
            "-o", "--output",
            help="出力ファイル名（省略時は config.txt の OUTPUT_FILE を使用）"
        )
        parser.add_argument(
            "--wait",
            action="store_true",
            help="終了時に Enter キー入力を待つ"
        )
        parser.add_argument(
            "--nowait",
            action="store_true",
            help="終了時に Enter キー入力を待たない"
        )
        args = parser.parse_args()

        # config.txt の読込
        cfg = load_kv_file(os.path.join(config.BASE_DIR, "config.txt"))
        config.output_level = int(cfg.get("OUTPUT_LEVEL", 1))
        st_pos_output_mode = int(cfg.get("ST_POS_OUTPUT_MODE", 1))
        tt_memory_mb = int(cfg.get("TT_MEMORY_MB", 256))
        cfg_input = cfg.get("INPUT_FILE", "")
        cfg_output = cfg.get("OUTPUT_FILE", "")
        if args.input:
            input_file = os.path.join(config.BASE_DIR, args.input)
        else:
            input_file = os.path.join(config.BASE_DIR, cfg_input)

        if args.output:
            output_file = os.path.join(config.BASE_DIR, args.output)
        else:
            output_file = os.path.join(config.BASE_DIR, cfg_output)
        if not input_file:
            raise ValueError("入力ファイルが指定されていません。")
        if not output_file:
            raise ValueError("出力ファイルが指定されていません。")
        

        # 出力先設定
        config.out_fp = open(output_file, "a", encoding="utf-8")

        # 入力ファイルの読込
        prob = load_kv_file(input_file)
        start_sfen = prob.get("START_SFEN", "")
        target_sfen = prob["TARGET_SFEN"]
        max_depth = int(prob["MAX_DEPTH"])
        limit = int(prob["LIMIT"])
        margin = int(prob["MARGIN"])
        fixed_rfs = set()
        if "FIXED_PIECES" in prob and prob["FIXED_PIECES"]:
            fixed_rfs = {int(x.strip()) for x in prob["FIXED_PIECES"].split(",")}
        if not target_sfen:
            raise ValueError(f"{input_file} に TARGET_SFEN が設定されていません。")
        if not max_depth:
            raise ValueError(f"{input_file} に MAX_DEPTH が設定されていません。")
        if max_depth <= 0:
            raise ValueError(f"{input_file} の MAX_DEPTH は 1 以上である必要があります。")
        if not limit:
            raise ValueError(f"{input_file} に LIMIT が設定されていません。")
        if margin < 0:
            raise ValueError(f"{input_file} の MARGIN は 0 以上である必要があります。")
        
        # デバッグ用
        debug_usis = load_debug_sol(input_file)
    except Exception as e:
        print("設定エラー", e)
        sys.exit(1)
    
    INI_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"  # 実戦初形
    if not start_sfen:
        start_sfen = INI_SFEN
    # 解数上限は1～10
    if limit > 10:
        limit = 10
    if limit < 1:
        limit = 1
    # 手待ちは0～5
    if margin > 5:
        margin = 5

    display_fixed_rfs = {}
    try:
        validate_sfen_has_king(start_sfen)
        start = cs.Board(start_sfen)
    except Exception as e:
        print("開始局面エラー", e)
        sys.exit(1)
    try:
        validate_sfen_has_king(target_sfen)
        target = cs.Board(target_sfen)
    except Exception as e:
        print("指定局面エラー", e)
        sys.exit(1)
    try:
        for x in fixed_rfs:
            r, f = validate_two_digits(x)
            p = start.piece(file_rank_to_sq(r, f))
            if p == 0:
                raise ValueError(f"{x}に駒がありません。")
            name = piece_value_to_name(p)
            side = name[:2]
            piece = name[-1]
            display_fixed_rfs[x] = f"{side}{x}{piece}"
    except Exception as e:
        print("不動駒設定エラー", e)
        sys.exit(1)
    
    dt_now = datetime.datetime.now()
    out('【開始】' + 'Structa ' + config.VERSION + ', ' + dt_now.strftime('%Y-%m-%d %H:%M:%S'), 0, console=True)
    out(f"入力ファイル：{input_file}", 1)
    out("開始局面：" + start.sfen(), 0, console=True)
    out("指定局面：" + target.sfen(), 0, console=True)
    if st_pos_output_mode == 2:
        text = "\n".join(get_boards_side_by_side(start, target))
    elif st_pos_output_mode == 1 and start.sfen() != INI_SFEN:
        text = "\n".join(get_boards_side_by_side(start, target))
    else:
        text = KIF.board_to_bod(target)
    out(text, 1, console=True)
    out("指定手数：" + str(max_depth), 0, console=True)
    out("解数上限：" + str(limit), 1, console=True)
    if display_fixed_rfs:
        s = "、".join(display_fixed_rfs.values())
        out(f"不動駒：{s}", 0, console=True)
    out('--------------------', 1, console=True)
    log_system_info()  # OUTPUT_LEVEL = 3 のときのみ環境情報を出力

    # 処理実行
    try:
        t0 = time.time()
        out("探索中…", 1, True, False)
        first_move_index = 0 # 再開機能実装時はここを変更する
        sols, stats, completed_first_moves, interrupted = find_all_paths_to_target(start, target, max_depth, limit, fixed_rfs, tt_memory_mb, margin, first_move_index, debug_usis)
        if interrupted:
            base_path = os.path.splitext(input_file)[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            resume_path = f"{base_path}_resume.json"
            resume_file = os.path.basename(resume_path)
            save_resume_file(resume_path, start, target, max_depth, limit, margin, fixed_rfs, completed_first_moves, sols)
            out("", 0, console=True, file=False)
            out(f"再開用ファイルを保存しました：{resume_file}", 0, console=True)
            out("【中断終了】", 0, console=True)
            out("", 0)
            raise KeyboardInterrupt
        elapsed = time.time() - t0

        out(f"検出解数：{len(sols)}", 0, console=True)
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        out(f"処理時間：{hours}時間{minutes}分{seconds}秒", 1, console=True)
        
        total = stats["total_nodes"]
        
        def pct(x):
            return f"{(x/total*100):.2f}%"
        
        out("---- 枝刈り統計 ----", 2)
        out(f"総ノード数  ：{total:,}", 1)
        out(
            f"盤上手数計算："
            f"{stats['pruned_need_moves']:,} ({pct(stats['pruned_need_moves'])})",
            2
        )
        out(
            f"先手持駒    ："
            f"{stats['pruned_diff_hand_s']:,} ({pct(stats['pruned_diff_hand_s'])})",
            2
        )
        out(
            f"後手持駒    ："
            f"{stats['pruned_diff_hand_g']:,} ({pct(stats['pruned_diff_hand_g'])})",
            2
        )

        out("---- 手数別 ----", 2)
        for d, c in enumerate(stats["pruned_by_depth"]):
            out(f"{d}手目での枝刈り：{c:,}", 2)

        out("---- 置換表 ----", 2)
        tt_lookups = stats.get("tt_lookups", 0)
        tt_hits = stats.get("tt_hits", 0)
        tt_stores = stats.get("tt_stores", 0)
        tt_updates = stats.get("tt_store_updates", 0)
        tt_evictions = stats.get("tt_evictions", 0)
        tt_size = stats.get("tt_size", 0)
        tt_max_size = stats.get("tt_max_size", 0)
        hit_rate = (tt_hits / tt_lookups * 100) if tt_lookups else 0.0
        out(f"参照回数    ：{tt_lookups:,}", 2)
        out(f"ヒット回数  ：{tt_hits:,}", 2)
        out(f"ヒット率    ：{hit_rate:.2f} %", 2)
        out(f"新規登録数  ：{tt_stores:,}", 2)
        out(f"更新回数    ：{tt_updates:,}", 2)
        out(f"追い出し回数：{tt_evictions:,}", 2)
        out(f"最終サイズ  ：{tt_size:,}", 2)
        out(f"登録数上限  ：{tt_max_size:,}", 2)
        out(f"メモリ上限  ：{tt_memory_mb:,} MB", 2)
        out(f"ヒット猶予  ：{margin}手", 2)
        out("---- コスト計算 TT ----", 3)
        cost_lookups = stats.get("cost_tt_lookups", 0)
        cost_hits = stats.get("cost_tt_hits", 0)
        cost_size = stats.get("cost_tt_size", 0)
        cost_max = stats.get("cost_tt_max_size", 0)
        hit_rate = (cost_hits / cost_lookups * 100) if cost_lookups else 0.0
        out(f"参照回数    ：{cost_lookups:,}", 3)
        out(f"ヒット回数  ：{cost_hits:,}", 3)
        out(f"ヒット率    ：{hit_rate:.2f} %", 3)
        out(f"最終サイズ  ：{cost_size:,}", 3)
        out(f"登録数上限  ：{cost_max:,}", 3)

        for idx, sol in enumerate(sols, 1):
            out(f"=== 解 #{idx} ===", 0)
            print_solution_kif(start, sol)

        dt_now = datetime.datetime.now()
        out('【終了】' + dt_now.strftime('%Y-%m-%d %H:%M:%S'), 0, console=True)
        out("", 0, console=True)
    except ValueError as e:
        print("入力値エラー:", e)
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            config.out_fp.close()
        except Exception:
            pass
    
    # 終了
    wait_exit = True
    if args.nowait:
        wait_exit = False
    elif args.wait:
        wait_exit = True
    if wait_exit:
        print("Enterキーで終了します。")
        try:
            input()
        except EOFError:
            pass
