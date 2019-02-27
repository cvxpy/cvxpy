import requests
from distutils.version import StrictVersion


def pypi_version(server):
    url = server + '/cvxpy/json'
    r = requests.get(url)
    data = r.json()
    versions = list(data["releases"].keys())
    versions.sort(key=StrictVersion)
    return versions[-1]


def conda_version(python_version, operating_system):
    #
    #   python_version must be one of {2.7, 3.5, 3.6, 3.7}
    #
    #   operating system must be one of {linux, osx, win}
    #
    pyvers_dict = {'2.7': 'py27', '3.5': 'py35', '3.6': 'py36', '3.7': 'py37'}
    pyvers = pyvers_dict[python_version]
    url = "https://api.anaconda.org/package/cvxgrp/cvxpy"
    r = requests.get(url)
    data = r.json()
    filestrings = [str(data['files'][i]['full_name']) for i in range(len(data['files']))]
    versions = []
    for fs in filestrings:
        fs = fs.split('/')
        # fs = ['cvxgrp', 'cvxpy', '<x.y.z>', '<os>', '<filename>' ]
        if operating_system in fs[3] and pyvers in fs[4]:
            versions.append(fs[2])
    versions.sort(key=StrictVersion)
    if len(versions) == 0:
        versions = ['0.0.0']
    return versions[-1]


def update_conda(python_version, operating_system):
    import cvxpy
    most_recent_remote = conda_version(python_version, operating_system)
    local_version = cvxpy.__version__
    return StrictVersion(local_version) > StrictVersion(most_recent_remote)

def update_pypi_wheel(python_version, operating_system, server):
    url = server + '/cvxpy/json'
    r = requests.get(url)
    data = r.json()
    relevant_versions = ['0.0.0']
    for version in data['releases']:
        if operating_system in data['releases'][version][0]['filename']:
            relevant_versions.append(version)
    relevant_versions.sort(key=StrictVersion)
    most_recent_remote = relevant_versions[-1]
    import cvxpy
    local_version = cvxpy.__version__
    return StrictVersion(local_version) > StrictVersion(most_recent_remote)
