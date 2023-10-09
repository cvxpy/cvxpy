"""
Copyright 2023, the CVXPY authors

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
import os
import subprocess

# IMPORTANT NOTE
#
#   Our versioning infrastructure is adapted from that used in SciPy v 1.9.0.
#   Specifically, much of this content came from
#   https://github.com/scipy/scipy/blob/91faf1ed4c3e83afe5009ffb7a9d18eab8dae683/tools/version_utils.py
#   It's possible that our adaptation has unnecessary complexity.
#   For example, SciPy might have certain contingencies in place for backwards
#   compatibilities that CVXPY does not guarantee.
#
#   Some comments in the SciPy source provide justification for individual code
#   snippets. We have mostly left those comments in-place, but we sometimes preface
#   them with the following remark:
#      "The comment below is from the SciPy code which we repurposed for cvxpy."
#

MAJOR = 1
MINOR = 4
MICRO = 0
IS_RELEASED = True
IS_RELEASE_BRANCH = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')[:7]

        # The comment below is from the SciPy code which we repurposed for cvxpy.
        #
        #   We need a version number that's regularly incrementing for newer commits,
        #   so the sort order in a wheelhouse of nightly builds is correct (see
        #   https://github.com/MacPython/scipy-wheels/issues/114). It should also be
        #   a reproducible version number, so don't rely on date/time but base it on
        #   commit history. This gives the commit count since the previous branch
        #   point from the current branch (assuming a full `git clone`, it may be
        #   less if `--depth` was used - commonly the default in CI):
        prev_version_tag = '^v{}.{}.0'.format(MAJOR, MINOR - 2)
        out = _minimal_ext_cmd(['git', 'rev-list', 'HEAD', prev_version_tag,
                                '--count'])
        COMMIT_COUNT = out.strip().decode('ascii')
        COMMIT_COUNT = '0' if not COMMIT_COUNT else COMMIT_COUNT
    except OSError:
        GIT_REVISION = "Unknown"
        COMMIT_COUNT = "Unknown"

    return GIT_REVISION, COMMIT_COUNT


def get_version_info():
    # The comment below is from the SciPy code which we adapted for cvxpy.
    #
    #   Adding the git rev number needs to be done inside
    #   write_version_py(), otherwise the import of cvxpy.version messes
    #   up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION, COMMIT_COUNT = git_version()
    elif os.path.exists('cvxpy/version.py'):
        # must be a source distribution, use existing version file
        # load it as a separate module to not load cvxpy/__init__.py
        import runpy
        ns = runpy.run_path('cvxpy/version.py')
        GIT_REVISION = ns['git_revision']
        COMMIT_COUNT = ns['git_revision']
    else:
        GIT_REVISION = "Unknown"
        COMMIT_COUNT = "Unknown"

    if not IS_RELEASED:
        FULLVERSION += '.dev0+' + COMMIT_COUNT + '.' + GIT_REVISION

    return FULLVERSION, GIT_REVISION, COMMIT_COUNT


def write_version_py(filename='cvxpy/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM CVXPY SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
commit_count = '%(commit_count)s'
release = %(isrelease)s
if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION, COMMIT_COUNT = get_version_info()

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'commit_count': COMMIT_COUNT,
                       'isrelease': str(IS_RELEASED)})
    finally:
        a.close()
