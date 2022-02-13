import builtins
import distutils.sysconfig
import distutils.version
import os
import platform
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

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
MINOR = 2
MICRO = 0
IS_RELEASED = False
IS_RELEASE_BRANCH = False
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


# BEFORE importing setuptools, remove MANIFEST. Otherwise it may not be
# properly updated when the contents of directories change (true for distutils,
# not sure about setuptools).
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

# The comment below is from the SciPy code which we adapted for cvxpy.
#
#   This is a bit hackish: we are setting a global variable so that the main
#   cvxpy __init__ can detect if it is being loaded by the setup routine, to
#   avoid attempting to load components that aren't built yet.  While ugly, it's
#   a lot more robust than what was previously being used.
builtins.__CVXPY_SETUP__ = True


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


# inject numpy headers
class build_ext_cvxpy(build_ext):
    def finalize_options(self) -> None:
        build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        # `__builtins__` can be a dict
        # see https://docs.python.org/2/reference/executionmodel.html
        if isinstance(__builtins__, dict):
            __builtins__['__NUMPY_SETUP__'] = False
        else:
            __builtins__.__NUMPY_SETUP__ = False

        import numpy
        self.include_dirs.append(numpy.get_include())


def is_platform_mac() -> bool:
    return sys.platform == 'darwin'


def not_on_windows(s: str) -> str:
    return s if platform.system().lower() != "windows" else ""


# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distutils behaviour which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.py. This behavior is
# motivated by Apple dropping support for libstdc++.
if is_platform_mac():
    if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
        current_system = distutils.version.LooseVersion(platform.mac_ver()[0])
        python_target = distutils.version.LooseVersion(
            distutils.sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET'))
        if python_target < '10.9' and current_system >= '10.9':
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

write_version_py()

# Optionally specify openmp flags when installing, eg
#
# CFLAGS="-fopenmp" LDFLAGS="-lgomp" python setup.py install
#
# TODO wheels should be compiled with openmp ...
canon = Extension(
    '_cvxcore',
    sources=['cvxpy/cvxcore/src/cvxcore.cpp',
             'cvxpy/cvxcore/src/LinOpOperations.cpp',
             'cvxpy/cvxcore/src/Utils.cpp',
             'cvxpy/cvxcore/python/cvxcore_wrap.cxx'],
    include_dirs=['cvxpy/cvxcore/src/',
                  'cvxpy/cvxcore/python/',
                  'cvxpy/cvxcore/include/'],
    extra_compile_args=[
        '-O3',
        '-std=c++11',
        '-Wall',
        '-pedantic',
        not_on_windows('-Wextra'),
        not_on_windows('-Wno-unused-parameter'),
    ],
    extra_link_args=['-O3'],
)

setup(
    name='cvxpy',
    version=str(VERSION),
    author='Steven Diamond, Eric Chu, Stephen Boyd',
    author_email='stevend2@stanford.edu, akshayka@cs.stanford.edu, '
                 'echu508@stanford.edu, boyd@stanford.edu',
    cmdclass={'build_ext': build_ext_cvxpy},
    ext_modules=[canon],
    packages=find_packages(exclude=["doc",
                                    "examples",
                                    "cvxpy.performance_tests"]),
    url='https://github.com/cvxpy/cvxpy',
    license='Apache License, Version 2.0',
    zip_safe=False,
    description='A domain-specific language for modeling convex optimization '
                'problems in Python.',
    package_data={
        'cvxpy': ['py.typed'],
    },
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    python_requires='>=3.6',
    install_requires=[
        "osqp >= 0.4.1",
        "ecos >= 2",
        "scs >= 1.1.6",
        "numpy >= 1.15",
        "scipy >= 1.1.0"
    ],
    setup_requires=["numpy >= 1.15"],
)
