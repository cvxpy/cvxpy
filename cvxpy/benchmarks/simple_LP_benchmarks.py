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

import numpy as np

import cvxpy as cp
from cvxpy.benchmarks.benchmark import Benchmark


class SimpleLPBenchmark(Benchmark):

    @staticmethod
    def name() -> str:
        return "Simple LP"

    @staticmethod
    def data_available(download_missing_data: bool) -> bool:
        return True

    @staticmethod
    def get_problem_instance() -> cp.Problem:
        n = int(1e7)
        c = np.arange(n)
        x = cp.Variable(n)
        objective = cp.Minimize(c @ x)
        constraints = [0 <= x, x <= 1]
        problem = cp.Problem(objective, constraints)
        return problem


class SimpleFullyParametrizedLPBenchmark(Benchmark):

    @staticmethod
    def name() -> str:
        return "Simple fully parametrized LP"

    @staticmethod
    def data_available(download_missing_data: bool) -> bool:
        return True

    @staticmethod
    def get_problem_instance() -> cp.Problem:
        n = int(1e4)
        p = cp.Parameter(n)
        x = cp.Variable(n)
        objective = cp.Minimize(p @ x)
        constraints = [0 <= x, x <= 1]
        problem = cp.Problem(objective, constraints)
        return problem


class SimpleScalarParametrizedLPBenchmark(Benchmark):

    @staticmethod
    def name() -> str:
        return "Simple scalar parametrized LP"

    @staticmethod
    def data_available(download_missing_data: bool) -> bool:
        return True

    @staticmethod
    def get_problem_instance() -> cp.Problem:
        n = int(1e6)
        p = cp.Parameter()
        c = np.arange(n)
        x = cp.Variable(n)
        objective = cp.Minimize((p * c)  @ x)
        constraints = [0 <= x, x <= 1]
        problem = cp.Problem(objective, constraints)
        return problem


if __name__ == '__main__':
    bench = SimpleLPBenchmark()
    bench.run_benchmark()
    bench.print_benchmark_results()

    bench = SimpleFullyParametrizedLPBenchmark()
    bench.run_benchmark()
    bench.print_benchmark_results()

    bench = SimpleFullyParametrizedLPBenchmark()
    bench.run_benchmark()
    bench.print_benchmark_results()
