import json
import urllib2
from distutils.version import StrictVersion


def version(test=False):
    if test:
        url = "https://test.pypi.org/pypi/cvxpy/json"
    else:
        url = "https://pypi.org/pypi/cvxpy/json"
    data = json.load(urllib2.urlopen(urllib2.Request(url)))
    versions = data["releases"].keys()
    versions.sort(key=StrictVersion)
    return versions[-1]


print version()
