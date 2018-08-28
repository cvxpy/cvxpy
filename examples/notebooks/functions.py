"""
Copyright 2013 Steven Diamond

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

# List of atoms for functions table in tutorial.
ATOM_DEFINITIONS = [
  {"name":"abs",
   "usage": "abs(x)",
   "meaning": "$ |x| $",
   "domain": "$ x \in \mathbf{R} $",
   "curvature": "Convex",
   "sign": "Positive",
   "monotonicity": ["Increasing for $ x \geq 0 $",
                    "Decreasing for $ x \leq 0 $"],
  },
  # {"name":"berhu",
  #  "arguments": ("Takes a single expression followed by a parameter as arguments. "
  #                "The parameter must be a positive number. "
  #                "The default value for the parameter is 1."),
  #  "meaning":
  #       ("\operatorname{berhu}(x,M) = \\begin{cases} |x| &\mbox{if } |x| \le M \\\\ "
  #        "\left(|x|^{2} + M^{2} \\right)/2M & \mbox{if } |x| > M \end{cases} \\\\"
  #        " \mbox{ where } x \in \mathbf{R}."),
  #  "curvature": "Convex",
  #  "sign": "Positive",
  #  "monotonicity": ["Increasing for $ x \geq 0 $",
  #                   "Decreasing for $ x \leq 0 $"],
  #  "example": "berhu(x, 1)",
  # },
  {"name":"entr",
   "usage": "entr(x)",
   "arguments": "Takes a single expression as an argument.",
   "meaning":
        ("$ \\begin{cases} -x \log (x) & x > 0 \\\\ "
         "0 & x = 0 \end{cases} \\\\ $"),

   "domain": "$ x \geq 0 $",
   "curvature": "Concave",
   "sign": "Unknown",
   "monotonicity": ["None"],
   "example": "entr(x)",
  },
  {"name":"exp",
   "usage": "exp(x)",
   "arguments": "Takes a single expression as an argument.",
   "meaning": "$ e^{x} $",
   "domain": "$ x \in \mathbf{R} $",
   "curvature": "Convex",
   "sign": "Positive",
   "monotonicity": ["Increasing"],
   "example": "exp(x)",
  },
  {"name":"geo_mean",
   "usage": "geo_mean(x1,...,xk)",
   "arguments": ("Takes a variable number of expressions as arguments. "
                 "These are interpreted as a vector."),
   "meaning": "$ (x_{1} \cdots x_{k})^{1/k} $",
   "domain": "$ x_{i} \geq 0 $",
   "curvature": "Concave",
   "sign": "Positive",
   "monotonicity": ["Increasing"],
   "example": "geo_mean(x, y)",
  },
  {"name":"huber",
   "usage": "huber(x)",
   "arguments": ("Takes a single expression followed by a parameter as arguments. "
                 "The parameter must be a positive number. "
                 "The default value for the parameter is 1."),
   "meaning":
        ("$ \\begin{cases} 2|x|-1 & |x| \ge 1 \\\\ "
         " |x|^{2} & |x| < 1 \end{cases} \\\\ $"),
   "domain": "$ x \in \mathbf{R} $",
   "curvature": "Convex",
   "sign": "Positive",
   "monotonicity": ["Increasing for $ x \geq 0 $",
                    "Decreasing for $ x \leq 0 $"],
   "cvx_equivalent": "huber, huber_pos, huber_circ",
  },
  {"name":"inv_pos",
   "usage": "inv_pos(x)",
   "arguments": "Takes a single expression as an argument.",
   "meaning": "$ 1/x $",
   "domain": "$ x > 0 $",
   "curvature": "Convex",
   "sign": "Positive",
   "monotonicity": ["Decreasing"],
   "example": "inv_pos(x)",
  },
  {"name":"kl_div",
   "usage": "kl_div(x,y)",
   "arguments": "Takes two expressions as arguments.",
   "meaning": "$ x \log (x/y)-x+y $",
   "domain": "$ x,y > 0 $",
   "curvature": "Convex",
   "sign": "Positive",
   "monotonicity": ["None"],
   "example": "kl_div(x, y)",
  },
  {"name":"log",
   "usage": "log(x)",
   "arguments": "Takes a single expression as an argument.",
   "meaning": "$ \log(x) $",
   "domain": "$ x > 0 $",
   "curvature": "Concave",
   "sign": "Unknown",
   "monotonicity": ["Increasing"],
   "example": "log(x)",
  },
  {"name":"log_sum_exp",
   "usage": "log_sum_exp(x1,...,xk)",
   "arguments": ("Takes a variable number of expressions as arguments. "
                 "These are interpreted as a vector."),
   "meaning": "$ \log \left(e^{x_{1}} + \cdots + e^{x_{k}} \\right) $",
   "domain": "$ x \in \mathbf{R}^{k} $",
   "curvature": "Convex",
   "sign": "Unknown",
   "monotonicity": ["Increasing"],
   "example": "log_sum_exp(x, y)",
  },
  {"name":"max",
   "usage": "max(x1,...,xk)",
   "arguments": ("Takes a variable number of expressions as arguments. "
                 "These are interpreted as a vector."),
   "meaning": "$ \max \left\{ x_{1}, \ldots , x_{k} \\right\} $",
   "domain": "$ x \in \mathbf{R}^{k} $",
   "curvature": "Convex",
   "sign": "max(sign(arguments))",
   "monotonicity": ["Increasing"],
   "example": "max(x, y)",
  },
  {"name":"min",
   "usage": "min(x1,...,xk)",
   "arguments": ("Takes a variable number of expressions as arguments. "
                 "These are interpreted as a vector."),
   "meaning": "$ \min \left\{ x_{1}, \ldots , x_{k} \\right\} $",
   "domain": "$ x \in \mathbf{R}^{k} $",
   "curvature": "Concave",
   "sign": "min(sign(arguments))",
   "monotonicity": ["Increasing"],
   "example": "min(x, y)",
  },
  # {"name":"norm",
  #  "arguments": ("Takes a variable number of expressions followed by a parameter as arguments. "
  #                "The expressions are interpreted as a vector. "
  #                "The parameter must either be a number p with p >= 1 or be Inf. "
  #                "The default parameter is 2."),
  #  "mathematical_definition": ("\\begin{aligned} "
  #                              " \operatorname{norm}(x,p) &= \left( \sum_{k=1}^{n} |x_{k}|^{p}} \\right)^{1/p} \\\\"
  #                              " \operatorname{norm}(x,\mbox{Inf}) &= \max \left\{ \left| x_{k} \\right| | k \in \{1,...,n \} \\right\} \\\\"
  #                              " \mbox{ where } x \in \mathbf{R}^{n}."
  #                              " \end{aligned} "),
  #  "curvature": "Convex",
  #  "sign": "Positive",
  #  "monotonicity": [("For all arguments], non-decreasing if the argument is positive"
  #                   " and non-increasing if the argument is negative.")],
  #  "example": "norm(x, y, 1)",
  # },
  {"name":"norm2",
   "usage": "norm2(x1,...,xk)",
   "arguments": ("Takes a variable number of expressions followed by a parameter as arguments. "
                 "The expressions are interpreted as a vector. "
                 "The parameter must either be a number p with p >= 1 or be Inf. "
                 "The default parameter is 2."),
   "meaning": ("$ \sqrt{x_{1}^{2} + \cdots + x_{k}^{2}} $"),
   "domain": "$ x \in \mathbf{R}^{k} $",
   "curvature": "Convex",
   "sign": "Positive",
   "monotonicity": ["Increasing for $ x \geq 0 $",
                    "Decreasing for $ x \leq 0 $"],
   "example": "norm(x, y, 1)",
  },
  {"name":"norm1",
   "usage": "norm1(x1,...,xk)",
   "arguments": ("Takes a variable number of expressions followed by a parameter as arguments. "
                 "The expressions are interpreted as a vector. "
                 "The parameter must either be a number p with p >= 1 or be Inf. "
                 "The default parameter is 2."),
   "meaning": ("$ |x_{1}| + \cdots + |x_{k}| $"),
   "domain": "$ x \in \mathbf{R}^{k} $",
   "curvature": "Convex",
   "sign": "Positive",
   "monotonicity": ["Increasing for $ x \geq 0 $",
                    "Decreasing for $ x \leq 0 $"],
   "example": "norm(x, y, 1)",
  },
  {"name":"norm_inf",
   "usage": "norm_inf(x1,...,xk)",
   "arguments": ("Takes a variable number of expressions followed by a parameter as arguments. "
                 "The expressions are interpreted as a vector. "
                 "The parameter must either be a number p with p >= 1 or be Inf. "
                 "The default parameter is 2."),
   "meaning": ("$ \max \left\{ |x_{1}|, \ldots, |x_{k}| \\right\} $"),
   "domain": "$ x \in \mathbf{R}^{k} $",
   "curvature": "Convex",
   "sign": "Positive",
   "monotonicity": ["Increasing for $ x \geq 0 $",
                    "Decreasing for $ x \leq 0 $"],
   "example": "norm(x, y, 1)",
  },
  {"name":"pos",
   "usage": "pos(x)",
   "arguments": "Takes a single expression as an argument.",
   "meaning": "$ \max \{x,0\} $",
   "domain": "$ x \in \mathbf{R}$",
   "curvature": "Convex",
   "sign": "Positive",
   "monotonicity": ["Increasing"],
   "example": "pos(x)",
  },
  {"name":"quad_over_lin",
   "usage": "quad_over_lin(x,y)",
   "arguments": "Takes two expressions as arguments.",
   "meaning": "$ x^{2}/y $",
   "domain": "$x \in \mathbf{R}$, y > 0",
   "curvature": "Convex",
   "sign": "Positive",
   "monotonicity": ["Increasing for $ x \geq 0 $",
                    "Decreasing for $ x \leq 0 $",
                    "Decreasing in y"],
   "example": "quad_over_lin(x, y)",
  },
  {"name":"sqrt",
   "usage": "sqrt(x)",
   "arguments": "Takes a single expression as an argument.",
   "meaning": "$ \sqrt{x} $",
   "domain": "$ x \geq 0 $",
   "curvature": "Concave",
   "sign": "Positive",
   "monotonicity": ["Increasing"],
   "example": "sqrt(x)",
  },
  {"name":"square",
   "usage": "square(x)",
   "arguments": "Takes a single expression as an argument.",
   "meaning": "$ x^{2} $",
   "domain": "$ x \in \mathbf{R} $",
   "curvature": "Convex",
   "sign": "Positive",
   "monotonicity": ["Increasing for $ x \geq 0 $",
                    "Decreasing for $ x \leq 0 $"],
   "cvx_equivalent": "square, square_pos, square_abs",
  },
  # {"name":"pow",
  #  "arguments": ("Takes a single expression followed by a parameter as arguments. "
  #                "The parameter must be a number. "),
  #  "mathematical_definition":
  #       ("\\begin{aligned} "
  #       " p &\le 0: \operatorname{pow}(x,p) &= "
  #       "\\begin{cases} x^{p} &\mbox{if } x > 0 \\\\"
  #       " +\infty &\mbox{if } x \le 0 \end{cases} \\\\"
  #       " 0 < p &< 1: \operatorname{pow}(x,p) &= "
  #       "\\begin{cases} x^{p} &\mbox{if } x \ge 0 \\\\"
  #       " -\infty &\mbox{if } x < 0 \end{cases}\\\\"
  #       " p &\ge 1: \operatorname{pow}(x,p) &= "
  #       "\\begin{cases} x^{p} &\mbox{if } x \ge 0 \\\\"
  #       " +\infty &\mbox{if } x < 0 \end{cases}\\\\"
  #       " \mbox{ where } x \in \mathbf{R}^{n}."
  #       " \end{aligned} "),
  #  "curvature": "Concave for 0 < p < 1. Otherwise convex.",
  #  "sign": "The argument's sign for 0 < p < 1. Otherwise positive.",
  #  "monotonicity": [("Non-increasing for p <= 0. Non-decreasing for 0 < p < 1. "
  #                   "If p >= 1, increasing for positive arguments and non-increasing for negative arguments.")],
  #  "example": "pow(x, 3)",
  #  "cvx_equivalent": "pow_p",
  # },
  {"name":"pow",
   "usage": "pow(x,p), $\\text{ } p \geq 1 $",
   "arguments": ("Takes a single expression followed by a parameter as arguments. "
                 "The parameter must be a number. "),
   "meaning": "$ x^{p} $",
   "domain": "$ x \geq 0 $",
   "curvature": "Convex",
   "sign": "Positive",
   "monotonicity": ["Increasing"],
   "example": "pow(x, 3)",
   "cvx_equivalent": "pow_p",
  },
  {"name":"pow",
   "usage": "pow(x,p), $\\text{ } 0 < p < 1 $",
   "arguments": ("Takes a single expression followed by a parameter as arguments. "
                 "The parameter must be a number. "),
   "meaning": "$ x^{p} $",
   "domain": "$ x \geq 0 $",
   "curvature": "Concave",
   "sign": "Positive",
   "monotonicity": ["Increasing"],
   "example": "pow(x, 3)",
   "cvx_equivalent": "pow_p",
  },
  {"name":"pow",
   "usage": "pow(x,p), $\\text{ } p \leq 0 $",
   "arguments": ("Takes a single expression followed by a parameter as arguments. "
                 "The parameter must be a number. "),
   "meaning": "$ x^{p} $",
   "domain": "$ x > 0 $",
   "curvature": "Convex",
   "sign": "Positive",
   "monotonicity": ["Decreasing"],
   "example": "pow(x, 3)",
   "cvx_equivalent": "pow_p",
  },
]
