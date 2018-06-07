import requests
from distutils.version import LooseVersion


def version(test=True):
    if test:
        url = "https://test.pypi.org/pypi/cvxpy/json"
    else:
        url = "https://pypi.org/pypi/cvxpy/json"
    r = requests.get(url)
    data = r.json()
    versions = data["releases"].keys()
    versions.sort(key=LooseVersion)
    return versions[-1]


print(version())
