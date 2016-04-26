"""
Copyright 2013 Steven Diamond, Eric Chu

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""
import os 
import sys
import nose

class cd():
    """Handling errors and returning to current folder"""
    
    def __init__(self,new_dir):
        self.new_dir = os.path.expanduser(new_dir)

    def __enter__(self):
        self.old_dir = os.getcwd()
        try:
            os.chdir(self.new_dir)
        except OSError:
            raise Exception("User does not have sufficient access to the cvxpy package path")	
        	
    def __exit__(self,etype,value,traceback):
        os.chdir(self.old_dir)


class tests():
    """Run all the tests from interpreter
    Functionality is equivalent to nosetests in base diretory"""

    def __init__(self):
    	"""Execute tests"""
        packages = sys.path

        cvxpy_location = filter(lambda x: 'cvxpy' in x,packages)
        if len(cvxpy_location) > 0:
    	    new_dir = cvxpy_location[0]
            with cd(new_dir):
                nose.run()
        else:
        	raise Exception("The CVXPY Package was not found")

