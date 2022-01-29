from packaging import version
import ast
import json
import os

versions = reversed(sorted(version.parse(v) for v in ast.literal_eval(os.environ["ALL_DEPLOYED_VERSIONS"])))
version_str = [str(v) for v in versions]

with open("versions.json", "w") as f:
    json.dump(version_str, f)
