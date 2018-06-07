import json
import urllib2
from distutils.version import LooseVersion


def version(test=True):
    if test:
        url = "https://test.pypi.org/pypi/cvxpy/json"
    else:
        url = "https://pypi.org/pypi/cvxpy/json"
    data = json.load(urllib2.urlopen(urllib2.Request(url)))
    versions = data["releases"].keys()
    versions.sort(key=LooseVersion)
    return versions[-1]


print(version())
