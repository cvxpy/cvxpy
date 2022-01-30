from packaging import version
import ast
import json
import os

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
