.. _labels:

Custom Labels
=============

You can assign custom labels to expressions and constraints to make 
debugging and model interpretation easier. This is especially useful for 
understanding complex models and improving code readability.

Basic Usage
-----------

Labels can be assigned to any expression or constraint using the ``set_label()`` method:

.. code:: python

    import cvxpy as cp
    import numpy as np
    
    # Create variables
    weights = cp.Variable(3, name="weights")
    
    # Create constraints with custom labels
    constraints = [
        (weights >= 0).set_label("non_negative_weights"),
        (cp.sum(weights) == 1).set_label("budget_constraint"),
        (weights <= 0.4).set_label("concentration_limits")
    ]
    
    # Labels appear when printing constraints
    for con in constraints:
        print(con)

::

    non_negative_weights: 0.0 <= weights
    budget_constraint: Sum(weights) == 1.0  
    concentration_limits: weights <= 0.4

Labels on Expressions
---------------------

For expressions, labels provide a way to name intermediate computations and 
build hierarchical models:

.. code:: python

    # Create expressions with custom labels  
    data = np.random.randn(3)
    
    # Level 1: Basic labeled components
    data_fit = cp.sum_squares(weights - data).set_label("data_fit")
    l2_reg = cp.norm(weights, 2).set_label("l2_regularization")
    l1_reg = cp.norm(weights, 1).set_label("l1_regularization")
    
    # Level 2: Compound expression with label
    # This shows how labels can be nested
    regularization = (l2_reg + 0.1 * l1_reg).set_label("total_regularization")
    
    # Level 3: Build objective from compound parts
    # Note: no label at this level to show recursion
    objective_expr = data_fit + 0.5 * regularization
    
    objective = cp.Minimize(objective_expr)
    
    # For expressions, use format_labeled() to see the labels
    print("Regularization:", regularization.format_labeled())
    print("Objective expression:", objective.format_labeled())
    
    # Create and display the full problem
    problem = cp.Problem(objective, constraints)
    print("\nFull problem with labels:")
    print(problem.format_labeled())

::

    Regularization: total_regularization
    Objective expression: minimize data_fit + 0.5 * total_regularization
    
    Full problem with labels:
    minimize data_fit + 0.5 * total_regularization
    subject to non_negative_weights: 0.0 <= weights
               budget_constraint: Sum(weights) == 1.0
               concentration_limits: weights <= 0.4

The ``set_label()`` method returns the object itself, allowing method chaining.
Custom labels are particularly helpful for debugging complex models and
improving code readability by providing meaningful names for mathematical expressions.

Note that constraints show their labels automatically when printed (``str()``),
while expressions require calling ``format_labeled()`` to see the labels. This
prevents labels from contaminating mathematical representations when
expressions are composed into larger expressions.

How format_labeled() Works
--------------------------

When ``format_labeled()`` formats an expression, it recursively walks down the expression
tree. For each sub-expression, if it has a label, it uses the label; otherwise it uses
the mathematical representation. When it encounters a labeled node, it stops recursion 
at that point. In the example above, ``total_regularization`` contains ``l2_regularization`` 
and ``l1_regularization`` internally, but ``format_labeled()`` displays only 
``total_regularization`` because it stops at that label rather than continuing deeper.

Dynamic Labels
--------------

Labels are "live" properties that can be modified after problem creation. This makes 
them useful for experimentation and debugging:

.. code:: python

    # Create and solve a problem
    x = cp.Variable()
    objective = cp.Minimize(cp.square(x) + cp.abs(x))
    problem = cp.Problem(objective)
    
    # Add labels after creation for clarity
    cp.square(x).set_label("quadratic_term")
    cp.abs(x).set_label("l1_penalty")
    
    print(problem.format_labeled())
    # Shows: minimize quadratic_term + l1_penalty
    
    # Labels can be changed or removed
    cp.square(x).label = "squared_loss"
    cp.abs(x).label = None  # Remove label
    
    print(problem.format_labeled())
    # Shows: minimize squared_loss + Abs(x)

.. warning::

   **Label Preservation and Expression Canonicalization**
   
   CVXPY's core functionality is to canonicalize expressions into standard forms that 
   solvers can handle. During this process, CVXPY may restructure your expression tree 
   through algebraic simplifications such as:
   
   - Flattening nested additions: ``(a + b) + c`` becomes ``a + b + c``
   - Converting subtractions to additions: ``(a - b) - c`` becomes ``a + (-b) + (-c)``
   - Combining like terms and other optimizations
   
   When these transformations occur, intermediate expression nodes may disappear, taking 
   their labels with them. For example::
   
       # This label may be lost due to flattening
       intermediate = (expr1 - expr2).set_label("intermediate_result")
       final = intermediate - expr3  # May become expr1 + (-expr2) + (-expr3)
   
   Labels work best on "leaf" expressions (variables, parameters, atomic functions) and 
   final compound expressions that won't be further composed. While ``format_labeled()`` 
   will show you the labels that survive canonicalization, be aware that some labels 
   on intermediate expressions may not appear in the final formatted output.