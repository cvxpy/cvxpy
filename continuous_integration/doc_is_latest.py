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

# This is a helper file to determine if the doc version that is currently deployed is the latest version

import argparse

from packaging import version

parser = argparse.ArgumentParser()
parser.add_argument('--c', type=str, help='current version')
parser.add_argument('--l', type=str, help='latest deployed version')
args = parser.parse_args()
if not args.l:
    print(True)
else:
    print(version.parse(args.c) >= version.parse(args.l))
