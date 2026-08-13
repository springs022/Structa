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
固定長・オープンアドレス方式の置換表。

従来は OrderedDict + LRU（move_to_end / popitem）で実装していたが、
    - 1 エントリあたりのメモリが 100〜200 バイト規模になる
    - 参照のたびに move_to_end のリンク付け替えコストが掛かる
という二重の負担があった。

ここでは array モジュールの固定長バッファを使い、
    - 到達不能置換表  : 9 バイト / エントリ（キー 8 + 残り手数 1）
    - コスト計算置換表 : 12 バイト / エントリ（キー 8 + 値 4）
としている。同じメモリ上限でおよそ 20 倍のエントリ数を保持できる。

置換方式は「2 スロットバケット + 残り手数優先」。
インデックス idx と idx^1 の 2 か所だけを見るので、
探索コストは常に定数（最悪 2 回の比較）である。
"""
from array import array

_EMPTY = 0          # vals の 0 は「空き」を表す（残り手数は +1 して格納する）
_COST_EMPTY = -1

# 到達不能置換表 1 エントリのバイト数（キー 8 + 値 1）
TT_ENTRY_SIZE = 9
# コスト計算置換表 1 エントリのバイト数（キー 8 + 値 4）
COST_TT_ENTRY_SIZE = 12

# コスト値のエンコード上限。avail（片側の残り手数）は現実的に 127 を超えないため、
# ここで丸めても「need > avail」の判定結果は変わらない。
_COST_CLAMP = 255


def _pow2_floor(n: int, minimum: int = 1024) -> int:
    """n 以下で最大の 2 のべき乗を返す（下限 minimum）。"""
    size = minimum
    while size * 2 <= n:
        size *= 2
    return size


class UnreachableTT:
    """
    「この局面は残り remain 手では目標局面に到達できない」を記録する置換表。

    stats は従来と同じキーを持つ dict を外から渡す。
    """

    __slots__ = ("keys", "vals", "mask", "size", "stats", "_count")

    def __init__(self, max_entries: int, stats: dict):
        self.size = _pow2_floor(max_entries)
        self.mask = self.size - 1
        # array の反復で直接確保する（bytes の一時オブジェクトを作らないため、
        # 初期化時のピークメモリが 2 倍にならない）
        self.keys = array("Q", [0]) * self.size
        self.vals = array("b", [0]) * self.size
        self.stats = stats
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def hit(self, h: int, remain: int, margin: int) -> bool:
        """
        従来の tt_hit と同じ判定。
        「同じ残り手数で失敗済み」または
        「margin より多い残り手数でも失敗していた」なら打ち切る。
        """
        stats = self.stats
        stats["lookups"] += 1
        idx = h & self.mask
        keys = self.keys
        vals = self.vals
        v = vals[idx]
        if v == _EMPTY or keys[idx] != h:
            idx ^= 1
            v = vals[idx]
            if v == _EMPTY or keys[idx] != h:
                return False
        delta = (v - 1) - remain
        if delta == 0 or delta > margin:
            stats["hits"] += 1
            return True
        return False

    def store(self, h: int, remain: int) -> None:
        stats = self.stats
        keys = self.keys
        vals = self.vals
        idx0 = h & self.mask
        idx1 = idx0 ^ 1
        v0 = vals[idx0]
        v1 = vals[idx1]

        # 既存エントリの更新
        if v0 != _EMPTY and keys[idx0] == h:
            if remain + 1 > v0:
                vals[idx0] = remain + 1
                stats["store_updates"] += 1
            return
        if v1 != _EMPTY and keys[idx1] == h:
            if remain + 1 > v1:
                vals[idx1] = remain + 1
                stats["store_updates"] += 1
            return

        # 空きスロットを優先
        if v0 == _EMPTY:
            idx = idx0
        elif v1 == _EMPTY:
            idx = idx1
        else:
            # どちらも埋まっているなら「残り手数が小さい方」を捨てる。
            # 残り手数が大きいエントリほど、作るのに掛かった部分木が大きい。
            idx = idx0 if v0 <= v1 else idx1
            if (vals[idx] - 1) > remain:
                # 捨てる価値のある新エントリではないので登録しない
                return
            stats["evictions"] += 1
            self._count -= 1
        keys[idx] = h
        vals[idx] = remain + 1
        stats["stores"] += 1
        self._count += 1


class CostTT:
    """盤上手数計算（need_s, need_g）のメモ用置換表。"""

    __slots__ = ("keys", "vals", "mask", "size", "stats", "_count")

    def __init__(self, max_entries: int, stats: dict):
        self.size = _pow2_floor(max_entries)
        self.mask = self.size - 1
        # array の反復で直接確保する（bytes の一時オブジェクトを作らないため、
        # 初期化時のピークメモリが 2 倍にならない）
        self.keys = array("Q", [0]) * self.size
        self.vals = array("i", [_COST_EMPTY]) * self.size
        self.stats = stats
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def get(self, h: int):
        """(need_s, need_g) を返す。未登録なら None。"""
        stats = self.stats
        stats["lookups"] += 1
        idx = h & self.mask
        keys = self.keys
        vals = self.vals
        v = vals[idx]
        if v == _COST_EMPTY or keys[idx] != h:
            idx ^= 1
            v = vals[idx]
            if v == _COST_EMPTY or keys[idx] != h:
                return None
        stats["hits"] += 1
        return (v >> 8), (v & 0xFF)

    def put(self, h: int, need_s: int, need_g: int) -> None:
        if need_s > _COST_CLAMP:
            need_s = _COST_CLAMP
        if need_g > _COST_CLAMP:
            need_g = _COST_CLAMP
        v = (need_s << 8) | need_g
        keys = self.keys
        vals = self.vals
        idx0 = h & self.mask
        idx1 = idx0 ^ 1
        if vals[idx0] == _COST_EMPTY:
            idx = idx0
            self._count += 1
        elif keys[idx0] == h:
            idx = idx0
        elif vals[idx1] == _COST_EMPTY:
            idx = idx1
            self._count += 1
        elif keys[idx1] == h:
            idx = idx1
        else:
            # 単純に片方を上書きする（コスト値は再計算可能なので損失は小さい）
            idx = idx0
        keys[idx] = h
        vals[idx] = v
