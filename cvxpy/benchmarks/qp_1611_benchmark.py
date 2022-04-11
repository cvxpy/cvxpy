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

import os
import shelve
import zipfile

import requests

import cvxpy as cp
from cvxpy.benchmarks.benchmark import Benchmark


class QP1611Benchmark(Benchmark):
    filename = os.path.join(Benchmark.data_folder, '1611/Problem.prob')
    zip_name = "Problem.prob.zip"
    url = "https://github.com/cvxpy/cvxpy/files/7922520/"

    @staticmethod
    def data_available(download_missing_data: bool) -> bool:
        if os.path.isfile(QP1611Benchmark.filename + ".dat"):
            return True
        local_zip = os.path.join(Benchmark.data_folder, QP1611Benchmark.zip_name)
        if download_missing_data:
            print(f"Downloading data for {QP1611Benchmark.name()} benchmark")
            with open(local_zip, "wb") as f:
                f.write(requests.get(QP1611Benchmark.url+QP1611Benchmark.zip_name).content)
            with zipfile.ZipFile(local_zip, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(Benchmark.data_folder, "1611"))
            os.remove(local_zip)
            return True
        return False

    @staticmethod
    def name() -> str:
        return "QP issue 1611"

    @staticmethod
    def get_problem_instance() -> cp.Problem:
        with shelve.open(QP1611Benchmark.filename) as ps:
            problem = ps['Problem']
        return problem

    @staticmethod
    def get_solver():
        return cp.ECOS_BB


if __name__ == '__main__':
    bench = QP1611Benchmark()
    bench.run_benchmark()
    bench.print_benchmark_results()
