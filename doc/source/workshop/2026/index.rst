.. _workshop-2026:

CVXPY Workshop 2026
===================

Overview
--------

The CVXPY Workshop brings together users and developers of CVXPY for tutorials,
talks, and discussions about convex optimization in Python.

Date & Location
---------------

* **Talks:** February 20, 2026
* **Hackathon:** February 21, 2026
* **Location:** Stanford University

Registration
------------

`Register here <https://docs.google.com/forms/d/e/1FAIpQLSexWIvIfNlpkY4KK3rw47qNg8MiYUT6vL-dPt1emACH6ChPzw/viewform?usp=header>`_

The Zoom link (for virtual participants) and room number (for in-person attendees) will be emailed to registered participants the day before the workshop.

Schedule
--------

Morning Session
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 15 85
   :header-rows: 0

   * - **9:00 AM**
     - **Introductory Remarks**

   * - **9:30 AM**
     - | **HiGHS for CVXPY** — Julian Hall *(Virtual)*
       |
       | HiGHS is the world's best open-source linear optimization software. Although it's been callable from CVXPY for a few years, its presence has grown, and it is the default MIP solver from CVXPY 1.8. This talk introduces HiGHS: its history, its solvers, its users, its people and its future.

   * - **10:00 AM**
     - | **Disciplined Biconvex Programming** — Hao Zhu *(Virtual)*
       |
       | We introduce disciplined biconvex programming (DBCP), a modeling framework for specifying and solving biconvex optimization problems. DBCP extends the principles of disciplined convex programming to biconvex problems, allowing users to specify biconvex optimization problems in a natural way based on a small number of syntax rules.

   * - **10:30 AM**
     - | **Optimal Matching with CVXPY** — Michael Howes *(In-Person)*
       |
       | The Stanford Statistics Department uses CVXPY to assign PhD students to courses to TA based on their preferences. The assignments are made by gathering the students preferences and then solving a linear program to find an optimal matching.

   * - **10:45 AM**
     - | **Adapting CVXPY for Optimization on Curved Spaces** — Nikhesh Kumar Murali *(Virtual)*
       |
       | We explore strategies for adapting and redefining notions of convexity for curved spaces, especially hyperbolic spaces with g-convexity and h-convexity.

   * - **11:00 AM**
     - | **Panel: Contributing to CVXPY** — William Zhang & Nikhil Devenathan *(In-Person)*

   * - **11:30 AM**
     - | **Talkback: CVXPY New Features** *(Hybrid)*

   * - **12:00 PM**
     - **Lunch Break**

Afternoon Session
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 15 85
   :header-rows: 0

   * - **1:00 PM**
     - | **A Tutorial on Graphs of Convex Sets** — Tobia Marcucci *(In-Person)*
       |
       | In this talk, I will give a tutorial on graphs of convex sets, with emphasis on their applications in robotics, planning, and decision making. I will present GCSOPT, an open-source Python library based on CVXPY that enables solving complex GCS problems in just a few lines of code.

   * - **1:30 PM**
     - | **Disciplined Nonlinear Programming** — Daniel Cederberg *(In-Person)*
       |
       | I introduce disciplined nonlinear programming (DNLP), a new syntax for specifying nonlinear programming problems. DNLP allows smooth functions to be freely mixed with nonsmooth convex and concave functions, with rules governing how the nonsmooth functions can be used.

   * - **2:00 PM**
     - | **pycvxset: Convex Sets in Python for Control and Analysis** — Abraham Vinod *(Virtual)*
       |
       | pycvxset is a new Python package to manipulate and visualize convex sets. We support polytopes and ellipsoids, and provide user-friendly methods to perform a variety of set operations.

   * - **2:15 PM**
     - | **Riskfolio-Lib: Advanced Portfolio Optimization** — Dany Cejas *(Virtual)*
       |
       | Advanced portfolio optimization with convex and integer programming in Python.

   * - **2:30 PM**
     - | **Talkback: Serialization Specifications** *(Hybrid)*

   * - **3:00 PM**
     - | **Minimizing Electricity-Related Costs and Emissions with EECO** — Fletcher Chapin *(In-Person)*
       |
       | We present Electric Emissions & Cost Optimizer (EECO), a software package for optimizing electricity-related emissions and costs. After discussing the technical details, we present a case study demonstrating how to use EECO with a CVXPY battery model.

   * - **3:15 PM**
     - | **Development Plans for CVXR** — Balasubramanian Narasimhan & Anqi Fu *(In-Person)*
       |
       | CVXR is the R port of CVXPY. In this talk we describe our plans for catching up with CVXPY features and present data from early experiments with new AI development tools.

   * - **3:30 PM**
     - | **Moreau: The Convex Optimization Solver for the Modern Workload** — Steven Diamond *(In-Person, Sponsored)*

   * - **3:45 PM**
     - | **Electrified Transportation Optimization with CVXPY** — Kyle Goodrick *(In-Person)*
       |
       | This presentation introduces a site optimization tool designed to identify the lowest-cost electrification strategy for shared transportation infrastructure serving multiple sectors.

   * - **4:00 PM**
     - | **MOCVXPY: A CVXPY Extension for Multiobjective Optimization** — Ludovic Salomon *(Virtual)*
       |
       | MOCVXPY is a library built on top of CVXPY for convex vector optimization. It enables practitioners to describe their convex vector optimization problems using an intuitive algebraic language.

   * - **4:15 PM**
     - | **Sponsored Talk** *(TBD)*

   * - **4:30 PM**
     - | **Keynote** — Stephen Boyd *(In-Person)*

   * - **5:00 PM**
     - | **Social** *(Location TBD)*

Hackathon (February 21)
-----------------------

Details coming soon.

Speakers
--------

Julian Hall
~~~~~~~~~~~

Julian Hall has a BA in Mathematics from the University of Oxford, a PhD from the University of Dundee, and since 1990 has been employed as a lecturer by the University of Edinburgh. His work has yielded journal articles that have won four best paper prizes. In 2016, with Ivet Galabova, he founded HiGHS, which has grown to be the world's best open-source linear optimization software.

Hao Zhu
~~~~~~~

Hao Zhu completed his bachelor's degree in computational chemistry at Nankai University in China, followed by a master's degree in neuroscience at the University of Freiburg, Germany. He is currently a PhD student in the Department of Computer Science at the University of Freiburg, jointly supervised by Prof. Joschka Boedecker and Prof. Moritz Diehl.

Michael Howes
~~~~~~~~~~~~~

Michael Howes is a fifth year graduate student in the Statistics Department at Stanford University. He works on the theory of auxiliary variables Markov Chain Monte Carlo algorithms and is advised by Prof. Persi Diaconis.

Tobia Marcucci
~~~~~~~~~~~~~~

Tobia Marcucci is an Assistant Professor in the Department of Electrical and Computer Engineering at UC Santa Barbara. He received his PhD in Computer Science from MIT under the supervision of Russ Tedrake and Pablo Parrilo. His doctoral dissertation was awarded the 2025 MIT EECS George M. Sprowls Thesis Award in Artificial Intelligence and Decision Making.

Daniel Cederberg
~~~~~~~~~~~~~~~~

Daniel Cederberg is a PhD student at Stanford University.

Abraham Vinod
~~~~~~~~~~~~~

Abraham P. Vinod is a Principal Research Scientist at Mitsubishi Electric Research Laboratories (MERL). He is the primary developer of pycvxset and co-developed SReachTools.

Dany Cejas
~~~~~~~~~~

Dany Cejas holds a BSc in Economic Engineering and Master in Finance. He is the creator and sole maintainer of Riskfolio-Lib (3,723 Github stars and +1.1M downloads) and author of "Advanced Portfolio Optimization: a Cutting-edge Quantitative Approach".

Fletcher Chapin
~~~~~~~~~~~~~~~

Fletcher Chapin is a 5th-year environmental engineering PhD candidate advised by Dr. Meagan Mauter in the WE3Lab at Stanford. His research focuses on data management solutions for the water sector and minimizing costs and emissions of wastewater treatment through flexible operation.

Balasubramanian Narasimhan
~~~~~~~~~~~~~~~~~~~~~~~~~~

Balasubramanian Narasimhan is a Senior Research Scientist in Departments of Biomedical Data Sciences and Statistics at Stanford. Together with Anqi Fu, he is the maintainer of CVXR, the R port of CVXPY.

Kyle Goodrick
~~~~~~~~~~~~~

Kyle Goodrick is a Senior Power Systems Research Engineer at ASPIRE, focusing on research at the intersection of electric power systems and electrified transportation. He earned his PhD in Electrical Engineering from the University of Colorado Boulder.

Steven Diamond
~~~~~~~~~~~~~~

Steven Diamond is the founder of CVXPY and its head maintainer. He is also a founder and COO of Optimal Intellect.

Ludovic Salomon
~~~~~~~~~~~~~~~

Ludovic Salomon is a researcher at Hydro-Québec. He holds a PhD in applied mathematics and computer science from Polytechnique Montréal. His research interests are in numerical optimization with a focus on multiobjective optimization.

Stephen Boyd
~~~~~~~~~~~~

Stephen P. Boyd is the Samsung Professor of Engineering and Professor of Electrical Engineering at Stanford University. His current research focus is on convex optimization applications in control, signal processing, machine learning, and finance. In 2017 he received the IEEE James H. Mulligan, Jr. Education Medal.

Contact
-------

For questions about the workshop, please email cvxpydevs@gmail.com.
