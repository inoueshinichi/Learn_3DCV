"""テスト用ユーティリティ関数
"""
import datetime

def test_elapsed_time(f):
    def _wrapper(*args, **kargs):

        print(f"{datetime.datetime.now()} [Start] {f.__name__}")

        v = f(*args, **kargs)

        print(f"{datetime.datetime.now()} [End] {f.__name__}")

        return v
    
    return _wrapper