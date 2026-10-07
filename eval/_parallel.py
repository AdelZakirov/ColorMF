"""Ordered CPU work with bounded prefetch and predictable error propagation."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice


def ordered_map(function, items, workers):
    """At most 2 * workers submitted items; consume in original input order."""
    if workers == 1:
        yield from map(function, items)
        return
    items = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque(pool.submit(function, item) for item in islice(items, workers * 2))
        try:
            while pending:
                future = pending.popleft()
                yield future.result()
                try:
                    item = next(items)
                except StopIteration:
                    continue
                pending.append(pool.submit(function, item))
        finally:
            for future in pending:
                future.cancel()
