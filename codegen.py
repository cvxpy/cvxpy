import cvxpy as cp
import numpy as np
import IPython as ipy

m, n = 20, 10
Aval = np.random.randn(m, n)
bval = np.random.randn(m)

x = cp.Variable(n)
A = cp.Parameter((m, n), value=Aval)
b = cp.Parameter(m, value=bval)
prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [cp.norm(x, 2) <= 1])

def codegen(prob, parameters, variables):
	data, _, _ = prob.get_problem_data(
                solver=cp.SCS)
	compiler = data[cp.settings.PARAM_PROB]
	param_ids = [p.id for p in parameters]
	param_vals = [p.value for p in parameters]

	# run canonicalization once to check that it works, and to reduce problem data tensor
	c, _, neg_A, b = compiler.apply_parameters(dict(zip(param_ids, param_vals)), keep_zeros=True)


	s = "import numpy as np\nfrom scipy import sparse\nimport ecos\n\n"
	param_args = ""
	for ide in param_ids[:-1]:
		param_args += "param%d, " % ide
	param_args += "param%d" % param_ids[-1]
	s += "def solve(%s):\n" % param_args

	return s
s = codegen(prob, [A, b], [x])

print(s)