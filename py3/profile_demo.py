

import cProfile


def factorial(n):
    rtv = 1
    for i in range(1, n+1):
        rtv *= i

    return rtv


with cProfile.Profile() as pr:

    n = factorial(10000)
    print(str(n)[:5])

    # ... do something ...

pr.print_stats()
