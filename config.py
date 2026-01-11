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
import sys
import os

VERSION = "1.0.0"
output_level = 1
out_fp = None

def get_base_dir():
    if getattr(sys, 'frozen', False):
        # exe 実行時
        return os.path.dirname(sys.executable)
    else:
        # python 実行時
        return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_dir()