# HACK: import coptpy first if its available because if we import it
# after cvxcore, the SWIG module initialization messes it up
try:
	import coptpy
except ImportError:
	pass
