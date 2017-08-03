from cvxpy import *
from os import listdir
import numpy
import glob
import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt



def run_testfile(filename):
	print filename


	settings.USE_CVXCANON = False	
	exec("from cvxpy import *\nsettings.USE_CVXCANON=False\n" + open(filename).read())
	oldtime = TIME
	OLD_ANSWERS = ANSWERS


	settings.USE_CVXCANON = True
	exec("from cvxpy import *\nsettings.USE_CVXCANON=True\n" + open(filename).read())
	newtime =  TIME
	NEW_ANSWERS = ANSWERS
	if(len(NEW_ANSWERS) != len(OLD_ANSWERS)):
		print "**** TEST " + filename + " FAILED: Different Number of Answers *****"
	for i in range(min(len(OLD_ANSWERS), len(NEW_ANSWERS))):
		if (OLD_ANSWERS[i] == -float('inf') and NEW_ANSWERS[i] != -float('inf')):
			print "**** TEST " + filename + " FAILED: Different Answers *****"
			return (oldtime, newtime)
		elif(OLD_ANSWERS[i] == float('inf') and NEW_ANSWERS[i] != float('inf')):
			print "**** TEST " + filename + " FAILED: Different Answers *****"
			return (oldtime, newtime)
		elif(OLD_ANSWERS[i] == NEW_ANSWERS[i]):
			continue
		elif(np.abs(OLD_ANSWERS[i] - NEW_ANSWERS[i]) > 1e-6 ):
			print "**** TEST " + filename + " FAILED: Different Answers *****"
			return (oldtime, newtime)
	

	print "***** TEST " + filename + " SUCCESSS, SAME ANSWERS ******"
	return (oldtime, newtime)

files = glob.glob("./364A_scripts/*.py")

oldtime = {};
newtime = {};
fnames = [];
iters = 1

for testfile in files:
	oldtime[testfile] = []
	newtime[testfile] = []
	
	for _ in range(iters):
		print "testing: ", testfile
		try:
			(o, n) = run_testfile(testfile);
			oldtime[testfile] += [o]
			newtime[testfile] += [n]
			fnames += [testfile]
		except (RuntimeError, TypeError, NameError, AttributeError):
			pass
list(set(fnames))
