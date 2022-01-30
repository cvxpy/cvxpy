"""
Copyright 2022 the CVXPY developers
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import ast
import json
import os

from packaging import version

# This script creates a "versions.json" file listing all currently deployed version of the documentation

if os.environ["ALL_DEPLOYED_VERSIONS"]:
    deployed_versions = [version.parse(v) for v in ast.literal_eval(os.environ["ALL_DEPLOYED_VERSIONS"])]
else:
    deployed_versions = []
this_version = version.parse(os.environ["CURRENT_VERSION"])

all_versions = {this_version} | set(deployed_versions)

versions = reversed(sorted(all_versions))
version_str = [str(v) for v in versions]

with open("versions.json", "w") as f:
    json.dump(version_str, f)
