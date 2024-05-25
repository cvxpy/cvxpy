# HACK: import coptpy first if its available because if we import it
# after cvxcore, the SWIG module initialization messes it up
try:
	import coptpy
except ImportError:
	pass

# TODO(akshayka): This is a hack; the swig-auto-generated cvxcore.py
# tries to import cvxcore as `from . import _cvxcore`
try:
	import _cvxcore
except ImportError:
	pass
