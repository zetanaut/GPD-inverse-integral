"""Exact-input cache for repeated trust-region coordinate-model construction.

Candidate optimization only. Radius is outside model_coordinates; any changed
R, g, A or setting invalidates the cache. The original function remains intact.
"""
import hashlib
import json
import time
import numpy as np


class ExactModelCache:
    def __init__(self, original):
        self.original = original
        self.key = None
        self.result = None
        self.stats = dict(calls=0, hits=0, misses=0, hashing_seconds=0., build_seconds=0.)

    def __call__(self, R, g, A, settings):
        self.stats['calls'] += 1
        start = time.perf_counter()
        h = hashlib.sha256(json.dumps(settings, sort_keys=True, allow_nan=False).encode())
        for a in (R, g, A):
            a = np.ascontiguousarray(a)
            h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
            if a.size:
                h.update(memoryview(a).cast('B'))
        key = h.digest()
        self.stats['hashing_seconds'] += time.perf_counter()-start
        if key != self.key:
            start = time.perf_counter()
            result = self.original(R, g, A, settings)
            self.stats['build_seconds'] += time.perf_counter()-start
            protected = {}
            for name, value in result.items():
                if isinstance(value, np.ndarray):
                    value = value.copy()
                    value.setflags(write=False)
                protected[name] = value
            self.result, self.key = protected, key
            self.stats['misses'] += 1
        else:
            self.stats['hits'] += 1
        # Caller can alter its mapping but not poison the saved arrays.
        return dict(self.result)


def install(solver):
    cache = ExactModelCache(solver.model_coordinates)
    solver.model_coordinates = cache
    return cache
