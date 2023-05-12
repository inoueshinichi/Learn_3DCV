"""テスト用ユーティリティ
"""

import os
import sys

import inspect
import time

from type_hint import *

def test_decorator(method) -> Any:

    def __wrapper(*args, **kwargs):
        # 前処理
        print(f"[START] {method.__name__}")
        tp_start = time.perf_counter_ns()

        ret = method(*args, **kwargs)

        tp_end = time.perf_counter_ns()

        # 後処理
        duration_ms = (tp_end - tp_start) / 1000
        print("[Time] {}[ms]".format(duration_ms))
        print(f"[END] {method.__name__}")

        return ret
    
    return __wrapper