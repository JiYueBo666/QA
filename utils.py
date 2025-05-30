import time
from functools import wraps


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # 开始计时
        result = func(*args, **kwargs)  # 执行原函数
        end_time = time.perf_counter()  # 结束计时
        elapsed_time = end_time - start_time
        print(f"函数 {func.__name__}{args} 耗时 {elapsed_time:.6f} 秒")
        return result

    return wrapper
