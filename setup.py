from distutils.core import setup

setup(
    name='cvxpy',
    version='0.1',
    author='Steven Diamond, Eric Chu, Stephen Boyd',
    author_email='stevend2@stanford.edu, echu508@stanford.edu, boyd@stanford.edu',
    packages=[  'cvxpy',
                'cvxpy.atoms',
                'cvxpy.constraints',
                'cvxpy.expressions',
                'cvxpy.interface',
                'cvxpy.problems',
                'cvxpy.tests'],
    package_dir={'cvxpy': 'cvxpy'},
        url='http://github.com/cvxgrp/cvxpy/',
    license='...',
    description='A domain-specific language for modeling convex optimization problems in Python.',
    long_description=open('README.md').read(),
    requires = ["cvxopt(>= 1.1.6)",
                "ecos(>=1.0)"] # this doesn't appear to do anything unfortunately
)
