import math
import time


def now_s():
    return time.perf_counter()


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    return ordered[f] * (c - k) + ordered[c] * (k - f)


def mean(values):
    if not values:
        return None
    return sum(values) / len(values)
