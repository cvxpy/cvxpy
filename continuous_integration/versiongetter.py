import requests
from pkg_resources import parse_version


def update_pypi_source(server: str) -> bool:
    # Gets the latest version on PyPi accompanied by a source distribution
    url = server + '/cvxpy/json'
    r = requests.get(url)
    if r.ok:
        data = r.json()
        releases = data["releases"]
        versions = [
            v for v in releases.keys() if 'sdist' in [rel['packagetype'] for rel in
                                                      releases[v]]]
        versions.sort(key=parse_version)
        remote_version = versions[-1]
        import cvxpy
        local_version = cvxpy.__version__
        return parse_version(local_version) > parse_version(remote_version)
    else:
        msg = 'The request to pypi returned status code' + str(r.status_code)
        raise RuntimeError(msg)


def map_runner_os_name_to_os(runner_os_name: str) -> str:
    if runner_os_name.lower() == 'linux':
        operating_system = 'linux'
    elif runner_os_name.lower() in {'osx', 'macos'}:
        operating_system = 'osx'
    elif runner_os_name.lower() == 'win':
        operating_system = 'win'
    else:
        raise Exception(f'Unknown runner_os {runner_os_name}.')
    return operating_system


def update_pypi_wheel(python_version: str, runner_os_name: str, server: str) -> bool:
    # python_version is expected to be
    #
    #   '2.7', '3.5', '3.6', '3.7', ... etc..
    #
    # runner_os_name is expected to be
    #
    #   'win' or 'osx' or 'macOS' or 'linux' or 'Linux'
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
    operating_system = map_runner_os_name_to_os(runner_os_name)
    if 'linux' in operating_system:
        return False
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
        relevant_versions.sort(key=parse_version)
        most_recent_remote = relevant_versions[-1]
        import cvxpy
        local_version = cvxpy.__version__
        return parse_version(local_version) > parse_version(most_recent_remote)
    else:
        msg = 'The request to pypi returned status code' + str(r.status_code)
        raise RuntimeError(msg)
