"""
Copyright 2025, the CVXPY Authors

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

CITATION_DICT = {}

CITATION_DICT["CVXPY"] = \
"""
@article{diamond2016cvxpy,
  author  = {Steven Diamond and Stephen Boyd},
  title   = {{CVXPY}: {A} {P}ython-embedded modeling language for convex optimization},
  journal = {Journal of Machine Learning Research},
  year    = {2016},
  volume  = {17},
  number  = {83},
  pages   = {1--5}
}

@article{agrawal2018rewriting,
  author  = {Agrawal, Akshay and Verschueren, Robin and Diamond, Steven and Boyd, Stephen},
  title   = {A rewriting system for convex optimization problems},
  journal = {Journal of Control and Decision},
  year    = {2018},
  volume  = {5},
  number  = {1},
  pages   = {42--60}
}
"""

CITATION_DICT["DCP"] = \
"""
@book{grant2006disciplined,
  title={Disciplined convex programming},
  author={Grant, Michael and Boyd, Stephen and Ye, Yinyu},
  year={2006},
  publisher={Springer}
}
"""

CITATION_DICT["DGP"] = \
"""
@article{agrawal2019disciplined,
  title={Disciplined geometric programming},
  author={Agrawal, Akshay and Diamond, Steven and Boyd, Stephen},
  journal={Optimization Letters},
  volume={13},
  pages={961--976},
  year={2019},
  publisher={Springer}
}
"""

CITATION_DICT["DQCP"] = \
"""
@article{agrawal2020disciplined,
  title={Disciplined quasiconvex programming},
  author={Agrawal, Akshay and Boyd, Stephen},
  journal={Optimization Letters},
  volume={14},
  number={7},
  pages={1643--1657},
  year={2020},
  publisher={Springer}
}
"""

CITATION_DICT["CVXOPT"] = \
"""
@article{vandenberghe2010cvxopt,
  title={The CVXOPT linear and quadratic cone program solvers},
  author={Vandenberghe, Lieven},
  journal={Online: http://cvxopt. org/documentation/coneprog. pdf},
  volume={53},
  year={2010}
}
"""

CITATION_DICT["GLPK"] = \
"""
@inproceedings{Oki2012GLPKL,
  title={GLPK (GNU Linear Programming Kit)},
  author={Eiji Oki},
  year={2012},
  url={https://api.semanticscholar.org/CorpusID:63578694}
}
"""

CITATION_DICT["GLPK"] = \
"""
@inproceedings{Oki2012GLPKL,
  title={GLPK (GNU Linear Programming Kit)},
  author={Eiji Oki},
  year={2012},
  url={https://api.semanticscholar.org/CorpusID:63578694}
}
"""

CITATION_DICT["GLPK_MI"] = \
"""
@inproceedings{Oki2012GLPKL,
  title={GLPK (GNU Linear Programming Kit)},
  author={Eiji Oki},
  year={2012},
  url={https://api.semanticscholar.org/CorpusID:63578694}
}
"""

CITATION_DICT["GLOP"] = \
"""
@software{ortools,
  title = {OR-Tools},
  version = { v9.11 },
  author = {Laurent Perron and Vincent Furnon},
  organization = {Google},
  url = {https://developers.google.com/optimization/},
  date = { 2024-05-07 }
}
"""

CITATION_DICT["CBC"] = \
"""
@inbook{Forrest2005,
  title = {CBC User Guide},
  ISBN = {9781877640216},
  url = {http://dx.doi.org/10.1287/educ.1053.0020},
  DOI = {10.1287/educ.1053.0020},
  booktitle = {Emerging Theory,  Methods,  and Applications},
  publisher = {INFORMS},
  author = {Forrest,  John and Lougee-Heimer,  Robin},
  year = {2005},
  month = sep,
  pages = {257–277}
}
"""

CITATION_DICT["COPT"] = \
"""
@article{ge2022cardinal,
  title={Cardinal Optimizer (COPT) user guide},
  author={Ge, Dongdong and Huangfu, Qi and Wang, Zizhuo and Wu, Jian and Ye, Yinyu},
  journal={arXiv preprint arXiv:2208.14314},
  year={2022}
}
"""

CITATION_DICT["ECOS"] = \
"""
@inproceedings{domahidi2013ecos,
  title={ECOS: An SOCP solver for embedded systems},
  author={Domahidi, Alexander and Chu, Eric and Boyd, Stephen},
  booktitle={2013 European control conference (ECC)},
  pages={3071--3076},
  year={2013},
  organization={IEEE}
}
"""

CITATION_DICT["ECOS_EXP"] = \
"""
@book{serrano2015algorithms,
  title={Algorithms for unsymmetric cone optimization and an implementation for problems
  with the exponential cone},
  author={Serrano, Santiago Akle},
  year={2015},
  publisher={Stanford University}
}
"""

CITATION_DICT["SCS"] = \
"""
@article{odonoghue2021operator,
  title={Operator splitting for a homogeneous embedding of the linear complementarity problem},
  author={O'Donoghue, Brendan},
  journal={SIAM Journal on Optimization},
  volume={31},
  number={3},
  pages={1999--2023},
  year={2021},
  publisher={SIAM}
}
"""

CITATION_DICT["SDPA"] = \
"""
@article{doi:10.1080/1055678031000118482,
    author    = {Yamashita, Makoto
                 and Fujisawa, Katsuki
                 and Kojima, Masakazu},
    title     = {Implementation and evaluation of SDPA 6.0 
    (Semidefinite Programming Algorithm 6.0)},
    journal   = {Optimization Methods and Software},
    volume    = {18},
    number    = {4},
    pages     = {491-505},
    year      = {2003},
    publisher = {Taylor & Francis},
    doi       = {10.1080/1055678031000118482},
    URL       = {https://doi.org/10.1080/1055678031000118482},
    eprint    = {https://doi.org/10.1080/1055678031000118482}
}

@Inbook{Yamashita2012,
    author    = {Yamashita, Makoto
                 and Fujisawa, Katsuki
                 and Fukuda, Mituhiro
                 and Kobayashi, Kazuhiro
                 and Nakata, Kazuhide
                 and Nakata, Maho},
    editor    = {Anjos, Miguel F.
                 and Lasserre, Jean B.},
    title           = {Latest Developments in the SDPA Family for Solving Large-Scale SDPs},
    bookTitle       = {Handbook on Semidefinite, Conic and Polynomial Optimization},
    year      = {2012},
    publisher = {Springer US},
    address   = {Boston, MA},
    pages     = {687--713},
    isbn      = {978-1-4614-0769-0},
    doi       = {10.1007/978-1-4614-0769-0_24},
    url       = {https://doi.org/10.1007/978-1-4614-0769-0_24}
}

@inproceedings{doi:10.1109/CACSD.2010.5612693,
    author    = {Nakata, Maho},
    booktitle = {2010 IEEE International Symposium on Computer-Aided Control System Design}, 
    title     = {A numerical evaluation of highly accurate multiple-precision arithmetic version
    of semidefinite programming solver: SDPA-GMP, -QD and -DD.}, 
    year      = {2010},
    volume    = {},
    number    = {},
    pages     = {29-34},
    doi       = {10.1109/CACSD.2010.5612693}
}

@article{Kim2011,
    author    = {Kim, Sunyoung
                 and Kojima, Masakazu
                 and Mevissen, Martin
                 and Yamashita, Makoto},
    title     = {Exploiting sparsity in linear and nonlinear matrix inequalities
    via positive semidefinite matrix completion},
    journal   = {Mathematical Programming},
    year      = {2011},
    month     = {Sep},
    day       = {01},
    volume    = {129},
    number    = {1},
    pages     = {33-68},
    issn      = {1436-4646},
    doi       = {10.1007/s10107-010-0402-6},
    url       = {https://doi.org/10.1007/s10107-010-0402-6}
}
"""

CITATION_DICT["DIFFCP"] = \
"""
@misc{agrawal2020differentiating,
    title={Differentiating Through a Cone Program}, 
    author={Akshay Agrawal and Shane Barratt and Stephen Boyd and Enzo Busseti and Walaa M. Moursi},
    year={2020},
    eprint={1904.09043},
    archivePrefix={arXiv},
    primaryClass={math.OC},
    url={https://arxiv.org/abs/1904.09043}, 
}
"""

CITATION_DICT["GUROBI"] = \
"""
@manual{gurobi,
  author = {Gurobi Optimization, LLC},
  title  = {Gurobi Optimizer Reference Manual},
  year   = 2025,
  url    = {https://www.gurobi.com}
}
"""

CITATION_DICT["OSQP"] = \
"""
@article{osqp,
  author  = {Stellato, B. and Banjac, G. and Goulart, P. and Bemporad, A. and Boyd, S.},
  title   = {{OSQP}: an operator splitting solver for quadratic programs},
  journal = {Mathematical Programming Computation},
  year    = {2020},
  volume  = {12},
  number  = {4},
  pages   = {637--672},
  doi     = {10.1007/s12532-020-00179-2},
  url     = {https://doi.org/10.1007/s12532-020-00179-2}
}
"""

CITATION_DICT["PIQP"] = \
"""
@inproceedings{schwan2023piqp,
  author    = {Schwan, Roland and Jiang, Yuning and Kuhn, Daniel and Jones, Colin N.},
  booktitle = {2023 62nd IEEE Conference on Decision and Control (CDC)},
  title     = {{PIQP}: A Proximal Interior-Point Quadratic Programming Solver},
  year      = {2023},
  volume    = {},
  number    = {},
  pages     = {1088-1093},
  doi       = {10.1109/CDC49753.2023.10383915}
}
"""

CITATION_DICT["PROXQP"] = \
"""
@inproceedings{bambade2022prox,
  title={Prox-qp: Yet another quadratic programming solver for robotics and beyond},
  author={Bambade, Antoine and El-Kazdadi, Sarah and Taylor, Adrien and Carpentier, Justin},
  booktitle={RSS 2022-Robotics: Science and Systems},
  year={2022}
}
"""

CITATION_DICT["QOCO"] = \
r"""
@misc{chari2025qoco,
  title         = {QOCO: A Quadratic Objective Conic Optimizer with Custom Solver Generation},
  author        = {Chari, Govind M and A{\c{c}}{\i}kme{\c{s}}e, Beh{\c{c}}et},
  year          = {2025},
  eprint        = {2503.12658},
  archiveprefix = {arXiv},
  primaryclass  = {math.OC},
  url           = {https://arxiv.org/abs/2503.12658}
}
"""

CITATION_DICT["CPLEX"] = \
"""
@article{manual1987ibm,
  title={Ibm ilog cplex optimization studio},
  author={Manual, CPLEX User’s},
  journal={Version},
  volume={12},
  number={1987-2018},
  pages={1},
  year={1987}
}
"""

CITATION_DICT["MOSEK"] = \
"""
@manual{mosek,
   author = {MOSEK ApS},
   title = {MOSEK Optimization Suite},
   year = {2025},
   url = {https://docs.mosek.com/latest/intro/index.html}
 }
"""

CITATION_DICT["MOSEK_EXP"] = \
"""
@article{Dahl2021,
  title = {A primal-dual interior-point algorithm for nonsymmetric exponential-cone optimization},
  volume = {194},
  ISSN = {1436-4646},
  url = {http://dx.doi.org/10.1007/s10107-021-01631-4},
  DOI = {10.1007/s10107-021-01631-4},
  number = {1–2},
  journal = {Mathematical Programming},
  publisher = {Springer Science and Business Media LLC},
  author = {Dahl,  Joachim and Andersen,  Erling D.},
  year = {2021},
  month = mar,
  pages = {341–370}
}
"""

CITATION_DICT["XPRESS"] = \
"""
@manual{xpress,
  author = {FICO},
  title  = {FICO Xpress Optimization},
  year   = {2025},
  url    = {https://www.fico.com/fico-xpress-optimization/docs/latest/overview.html}
}
"""

CITATION_DICT["NAG"] = \
"""
@misc{xpress,
  author = {NAG},
  title  = {Optimization Modelling Suite},
  year   = 2025,
  url    = {https://nag.com/mathematical-optimization/}
}
"""


CITATION_DICT["PDLP"] = \
"""
@article{applegate2021practical,
  title={Practical large-scale linear programming using primal-dual hybrid gradient},
  author={Applegate, David and D{\'\\i}az, Mateo and Hinder, Oliver and Lu, Haihao and Lubin, Miles
  and O'Donoghue, Brendan and Schudy, Warren},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={20243--20257},
  year={2021}
}

@article{applegate2025pdlp,
  title={PDLP: A Practical First-Order Method for Large-Scale Linear Programming},
  author={Applegate, David and D{\'\\i}az, Mateo and Hinder, Oliver and Lu, Haihao and Lubin, Miles
  and O'Donoghue, Brendan and Schudy, Warren},
  journal={arXiv preprint arXiv:2501.07018},
  year={2025}
}
"""

CITATION_DICT["SCIP"] = \
"""
@article{bolusani2024scip,
  title={The SCIP optimization suite 9.0},
  author={Bolusani, Suresh and Besan{\\c{c}}on, Mathieu and Bestuzheva, Ksenia
  and Chmiela, Antonia and Dion{\'\\i}sio, Jo{\\~a}o and Donkiewicz, Tim
  and van Doornmalen, Jasper and Eifler, Leon and Ghannam, Mohammed and Gleixner, Ambros
  and others},
  journal={arXiv preprint arXiv:2402.17702},
  year={2024}
}
"""

CITATION_DICT["SCIPY"] = \
"""
@ARTICLE{2020SciPy-NMeth,
  author  = {Virtanen, Pauli and Gommers, Ralf and Oliphant, Travis E. and
            Haberland, Matt and Reddy, Tyler and Cournapeau, David and
            Burovski, Evgeni and Peterson, Pearu and Weckesser, Warren and
            Bright, Jonathan and {van der Walt}, St{\'e}fan J. and
            Brett, Matthew and Wilson, Joshua and Millman, K. Jarrod and
            Mayorov, Nikolay and Nelson, Andrew R. J. and Jones, Eric and
            Kern, Robert and Larson, Eric and Carey, C J and
            Polat, {\\.I}lhan and Feng, Yu and Moore, Eric W. and
            {VanderPlas}, Jake and Laxalde, Denis and Perktold, Josef and
            Cimrman, Robert and Henriksen, Ian and Quintero, E. A. and
            Harris, Charles R. and Archibald, Anne M. and
            Ribeiro, Ant{\\^o}nio H. and Pedregosa, Fabian and
            {van Mulbregt}, Paul and {SciPy 1.0 Contributors}},
  title   = {{{SciPy} 1.0: Fundamental Algorithms for Scientific
            Computing in Python}},
  journal = {Nature Methods},
  year    = {2020},
  volume  = {17},
  pages   = {261--272},
  adsurl  = {https://rdcu.be/b08Wh},
  doi     = {10.1038/s41592-019-0686-2},
}
"""

CITATION_DICT["CLARABEL"] = \
"""
@misc{Clarabel_2024,
      title={Clarabel: An interior-point solver for conic programs with quadratic objectives}, 
      author={Paul J. Goulart and Yuwen Chen},
      year={2024},
      eprint={2405.12762},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
"""

CITATION_DICT["CUCLARABEL"] = \
"""
@misc{CuClarabel,
      title={CuClarabel: GPU Acceleration for a Conic Optimization Solver}, 
      author={Yuwen Chen and Danny Tse and Parth Nobel and Paul Goulart and Stephen Boyd},
      year={2024},
      eprint={2412.19027},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2412.19027}, 
}
"""

CITATION_DICT["DAQP"] = \
"""
@article{arnstrom2022dual,
  title={A Dual Active-Set Solver for Embedded Quadratic Programming Using Recursive
  LDL $\\^{}$\\{$T$\\}$ $ Updates},
  author={Arnstr{\"o}m, Daniel and Bemporad, Alberto and Axehill, Daniel},
  journal={IEEE Transactions on Automatic Control},
  volume={67},
  number={8},
  pages={4362--4369},
  year={2022},
  publisher={IEEE}
}
"""

CITATION_DICT["HIGHS"] = \
"""
@article{Huangfu2017,
  title = {Parallelizing the dual revised simplex method},
  volume = {10},
  ISSN = {1867-2957},
  url = {http://dx.doi.org/10.1007/s12532-017-0130-5},
  DOI = {10.1007/s12532-017-0130-5},
  number = {1},
  journal = {Mathematical Programming Computation},
  publisher = {Springer Science and Business Media LLC},
  author = {Huangfu,  Q. and Hall,  J. A. J.},
  year = {2017},
  month = dec,
  pages = {119–142}
}
"""

CITATION_DICT["MPAX"] = \
"""
@article{lu2024mpax,
  title={MPAX: Mathematical Programming in JAX},
  author={Lu, Haihao and Peng, Zedong and Yang, Jinwen},
  journal={arXiv preprint arXiv:2412.09734},
  year={2024}
}
"""

CITATION_DICT["CUOPT"] = \
"""
@software{cuOpt,
  title = {cuOpt},
  version = { 25.05 },
  organization = {NVIDIA},
  url = {https://docs.nvidia.com/cuopt/index.html},
  date = { 2025-05-29 }
}
"""
