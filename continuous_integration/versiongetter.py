import requests
from distutils.version import LooseVersion


def update_pypi_source(server):
    # Gets the latest version on PyPi accompanied by a source distribution
    url = server + '/cvxpy/json'
    r = requests.get(url)
    if r.ok:
        data = r.json()
        releases = data["releases"]
        versions = [
            v for v in releases.keys() if 'sdist' in [rel['packagetype'] for rel in
                                                      releases[v]]]
        versions.sort(key=LooseVersion)
        remote_version = versions[-1]
        import cvxpy
        local_version = cvxpy.__version__
        return LooseVersion(local_version) > LooseVersion(remote_version)
    else:
        msg = 'The request to pypi returned status code' + str(r.status_code)
        raise RuntimeError(msg)


def conda_version(python_version, operating_system):
    #
    #   python_version must be one of {2.7, 3.5, 3.6, 3.7}
    #
    #   operating system must be one of {linux, osx, win}
    #
    major_minor = python_version.split('.')
    pyvers = 'py' + major_minor[0] + major_minor[1]
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
    versions.sort(key=LooseVersion)
    if len(versions) == 0:
        versions = ['0.0.0']
    return versions[-1]


def update_conda(python_version, operating_system):
    import cvxpy
    most_recent_remote = conda_version(python_version, operating_system)
    local_version = cvxpy.__version__
    return LooseVersion(local_version) > LooseVersion(most_recent_remote)


def update_pypi_wheel(python_version, operating_system, server):
    # python_version is expected to be
    #
    #   '2.7', '3.5', '3.6', '3.7', ... etc..
    #
    # operating system is expected to be
    #
    #   'win' or 'osx' or 'linux'
    #
    # server is expected to be
    #
    #   'https://pypi.org/pypi' or 'https://test.pypi.org/pypi'
    #
    # our wheels are associated with an operating system and a python version.
    # We need to check the file names of existing wheels on pypi to see how the
    # current version of cvxpy compares to versions with the desired OS and
    # desired python version.
    #
    # Here is an example filename in the official pypi format:
    #
    #   cvxpy-1.0.24-cp36-cp36m-win_amd64.whl
    #
    # So we can check that the wheel contains the string given by "operating_system".
    # Checking that the file has a desired pythonversion can be done with a string
    # 'cp[MAJOR_VERSION][minor_version]' -- without the brackets.
    url = server + '/cvxpy/json'
    r = requests.get(url)
    major_minor = python_version.split('.')
    py_ver = 'cp' + major_minor[0] + major_minor[1]
    if 'linux' in operating_system:
        operating_system = 'manylinux'
    if r.ok:
        data = r.json()
        relevant_versions = ['0.0.0']
        for version, version_data in data['releases'].items():
            # version is something like '1.0.24'
            #
            # version_data is a list of dicts, with one dict
            # for each file (of this cvxpy version) hosted on pypi.
            #
            # pypi hosts source distributions, wheels, and eggs.
            filenames = [file_data['filename'] for file_data in version_data]
            for fn in filenames:
                if py_ver in fn and operating_system in fn:
                    relevant_versions.append(version)
        relevant_versions.sort(key=LooseVersion)
        most_recent_remote = relevant_versions[-1]
        import cvxpy
        local_version = cvxpy.__version__
        return LooseVersion(local_version) > LooseVersion(most_recent_remote)
    else:
        msg = 'The request to pypi returned status code' + str(r.status_code)
        raise RuntimeError(msg)
