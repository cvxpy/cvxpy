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
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.benchmarks.benchmark import Benchmark


class CVaRBenchmark(Benchmark):

    @staticmethod
    def name() -> str:
        return "CVaR"

    @staticmethod
    def data_available(download_missing_data: bool) -> bool:
        return True

    @staticmethod
    def get_problem_instance() -> cp.Problem:
        # Replaced real data with random values
        np.random.seed(0)
        price_scenarios = np.random.randn(131072, 192)
        forward_price_scenarios = np.random.randn(131072, 192)
        asset_energy_limits = np.random.randn(192, 2)
        bid_curve_prices = np.random.randn(192, 3)
        cvar_prob = 0.95
        cvar_kappa = 2.0

        num_scenarios, num_assets = price_scenarios.shape
        num_energy_segments = bid_curve_prices.shape[1] + 1

        price_segments = np.sum(
            forward_price_scenarios[:, :, None] > bid_curve_prices[None], axis=2
        )
        price_segments_flat = (
            price_segments + np.arange(num_assets) * num_energy_segments
        ).reshape(-1)
        price_segments_sp = sp.coo_matrix(
            (
                np.ones(num_scenarios * num_assets),
                (np.arange(num_scenarios * num_assets), price_segments_flat),
            ),
            shape=(num_scenarios * num_assets, num_assets * num_energy_segments),
        )

        prices_flat = (price_scenarios - forward_price_scenarios).reshape(-1)
        scenario_sum = sp.coo_matrix(
            (
                np.ones(num_scenarios * num_assets),
                (
                    np.repeat(np.arange(num_scenarios), num_assets),
                    np.arange(num_scenarios * num_assets),
                ),
            )
        )

        A = np.asarray((scenario_sum @ sp.diags(prices_flat) @ price_segments_sp).todense())
        c = np.mean(A, axis=0)
        gamma = 1.0 / (1.0 - cvar_prob) / num_scenarios
        kappa = cvar_kappa
        x_min = np.tile(asset_energy_limits[:, 0:1], (1, num_energy_segments)).reshape(-1)
        x_max = np.tile(asset_energy_limits[:, 1:2], (1, num_energy_segments)).reshape(-1)

        alpha = cp.Variable()
        x = cp.Variable(num_assets * num_energy_segments)

        problem = cp.Problem(
            cp.Minimize(c.T @ x),
            [
                alpha + gamma * cp.sum(cp.pos(A @ x - alpha)) <= kappa,
                x >= x_min,
                x <= x_max,
            ])

        return problem
