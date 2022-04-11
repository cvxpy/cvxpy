"""
Copyright, the CVXPY authors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import time
import tracemalloc
from abc import ABC, abstractmethod
from typing import Callable

import cvxpy as cp


class Benchmark(ABC):

    data_folder = "benchmark_data/"

    def __init__(self):
        self._timing = None
        self._memory_peak = None

    @staticmethod
    @abstractmethod
    def name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_problem_instance() -> cp.Problem:
        pass

    def run_benchmark(self) -> None:
        print(f"Running benchmark {self.name()}")
        timing, memory_peak = self.measure_time_and_memory(self.compile_problem)
        self.timing = timing
        self.memory_peak = memory_peak

    def compile_problem(self):
        prob = self.get_problem_instance()
        prob.get_problem_data(solver=self.get_solver())

    @staticmethod
    def measure_time_and_memory(func: Callable):
        tracemalloc.start()
        start = time.time()
        func()
        timing = time.time() - start
        _, memory_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_peak = memory_peak / 1024 ** 3  # bytes to GiB
        return timing, memory_peak

    def print_benchmark_results(self):
        print(self.name())
        print(f"Timing: {self.timing:.2f}s")
        print(f"Peak memory usage: {self.memory_peak:.2f}GiB\n\n")

    @property
    def timing(self):
        assert self._timing is not None
        return self._timing

    @timing.setter
    def timing(self, value):
        self._timing = value

    @property
    def memory_peak(self):
        assert self._memory_peak is not None
        return self._memory_peak

    @memory_peak.setter
    def memory_peak(self, value):
        self._memory_peak = value

    @staticmethod
    def get_solver():
        return cp.SCS

    @staticmethod
    @abstractmethod
    def data_available(download_missing_data: bool) -> bool:
        pass
