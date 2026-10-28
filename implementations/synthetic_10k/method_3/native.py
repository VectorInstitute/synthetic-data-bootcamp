"""Load original pure function bodies without running CLI imports or logging.

Only literal constants and function definitions are compiled. The original source
is read, never edited; no sys.modules replacements or monkeypatching are involved.
"""
import ast
import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import statistics
import string
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
# The first two methods stay in their existing location and are read-only inputs.
ROOT = PACKAGE_DIR.parent


@lru_cache(maxsize=None)
def native_functions(relative):
    path = ROOT / relative
    parsed = ast.parse(path.read_text(encoding='utf-8'))
    kept = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    for node in parsed.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kept.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            try:
                ast.literal_eval(node.value)
                kept.append(node)
            except (ValueError, TypeError):
                pass
    namespace = dict(globals(), logger=logging.getLogger('method_3.native'),
                     quantiles=statistics.quantiles)
    exec(compile(ast.fix_missing_locations(ast.Module(body=kept, type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for name in ('method_1', 'method_2')
            for p in sorted((ROOT / name).rglob('*.py'))}
