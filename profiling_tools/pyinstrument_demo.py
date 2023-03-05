import numpy as np

def add(n1: int, n2: int) -> int:
    return n1 + n2

def main():
    a = np.random.randn(10000).reshape((100,100))
    b = np.random.randn(20000).reshape((100,200))
    for _ in range(10000):
        c = a.dot(b)


def demo_pyinstrument_in_code():
    from pyinstrument import Profiler
    profiler = Profiler()
    profiler.start()

    a = np.random.randn(10000).reshape((100,100))
    b = np.random.randn(20000).reshape((100,200))
    for _ in range(10000):
        c = a.dot(b)

    profiler.stop()
    profiler.print()
if __name__ == '__main__':
    main()
    # demo_pyinstrument_in_code()

# run "pyinstrument pyinstrument_demo.py"
# pyinstrument --show-all pyinstrument_demo.py
# or wrap your code with profiler start, stop
