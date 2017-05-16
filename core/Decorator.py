import time


def time_it(cb):
    def wrapper():
        start_time = time.clock()
        cb()
        end_time = time.clock()
        seconds = end_time - start_time
        print("当前耗时", seconds)
    return wrapper
