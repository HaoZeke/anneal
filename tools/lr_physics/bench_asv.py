"""asv benchmark the config in this directory can import and run."""

import math


def quantile7(values, p):
    xs = sorted(values)
    h = (len(xs) - 1) * p
    lo = math.floor(h)
    hi = math.ceil(h)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (h - lo) * (xs[hi] - xs[lo])


class Quantile:
    def setup(self):
        self.samples = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]

    def time_median(self):
        quantile7(self.samples, 0.5)

    def time_iqr(self):
        quantile7(self.samples, 0.75) - quantile7(self.samples, 0.25)
