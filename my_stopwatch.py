# my_stopwatch.py
import time
from contextlib import contextmanager

# FROM: GPT-4


class Stopwatch:
    def __init__(self):
        self.reset()

    @contextmanager
    def __call__(self, name):
        if self.start_time is not None:
            raise ValueError(
                f"Nested use of stopwatch is not allowed. {self.current_name} is still running."
            )
        self.current_name = name
        self.start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        elapsed_time = end_time - self.start_time
        self.timers[name] = self.timers.get(name, 0) + elapsed_time
        self.start_time = None

    def __str__(self):
        max_name_len = max(len(name) for name in self.timers.keys())
        result = []
        for name, elapsed in sorted(self.timers.items(), key=lambda x: -x[1]):
            result.append(f"{name.ljust(max_name_len)} : {elapsed:.6f} seconds")
        return "Timers\n\n" + "\n".join(result) + "\n----------\n"

    def reset(self):
        self.timers = {}
        self.start_time = None
        self.current_name = None


stopwatch = Stopwatch()

if __name__ == "__main__":
    with stopwatch("sleep"):
        time.sleep(1)
    with stopwatch("sleep"):
        time.sleep(2)
    with stopwatch("sleep 2"):
        time.sleep(4)
    print(stopwatch)
