.. _workshop-2026:

.. raw:: html

   <style>
   details.sd-dropdown > summary { list-style: none; }
   details.sd-dropdown > summary::-webkit-details-marker { display: none; }
   details.sd-dropdown > summary::marker { display: none; }
   details.sd-dropdown > summary::before { display: none !important; }
   .sd-summary-state-marker { display: none !important; }
   .sd-summary-text { margin-right: 0.5em; }
   </style>

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
* **Location:** `CoDa E160 <https://maps.app.goo.gl/ANfH9ZehwgvSNnS76>`_, Stanford University

Registration
------------

`Register here <https://docs.google.com/forms/d/e/1FAIpQLSexWIvIfNlpkY4KK3rw47qNg8MiYUT6vL-dPt1emACH6ChPzw/viewform?usp=header>`_

The Zoom link (for virtual participants) will be emailed to registered participants the day before the workshop.

Schedule
--------

All times are in Pacific Time (UTC−8).

.. raw:: html

   <style>
   #tz-toggle-btn {
     display: none;
     margin-bottom: 1em;
     padding: 0.4em 1.2em;
     cursor: pointer;
     border: 1px solid var(--md-default-fg-color--lightest, #ccc);
     border-radius: 4px;
     background: var(--md-default-bg-color, #f5f5f5);
     color: var(--md-default-fg-color, #333);
     font-size: 0.95em;
   }
   #tz-toggle-btn:hover {
     background: var(--md-accent-fg-color--transparent, rgba(0,0,0,0.05));
   }
   </style>
   <button id="tz-toggle-btn" aria-label="Toggle timezone display" aria-pressed="false"></button>
   <script>
   (function() {
     var userTZ = Intl.DateTimeFormat().resolvedOptions().timeZone;
     var showingLocal = false;
     var timeEls = [];
     // Compare formatted times to detect whether the user is in Pacific Time.
     var testDate = new Date('2026-02-20T12:00:00-08:00');
     var pacificStr = testDate.toLocaleString('en-US', {timeZone: 'America/Los_Angeles', hour: 'numeric', minute: '2-digit', hour12: true});
     var userStr = testDate.toLocaleString('en-US', {timeZone: userTZ, hour: 'numeric', minute: '2-digit', hour12: true});
     var isSameTZ = (pacificStr === userStr);

     function findTimeElements() {
       // Scope search to schedule list-tables to avoid matching unrelated elements.
       var tables = document.querySelectorAll('.table-wrapper, table.docutils');
       var re = /^(\d{1,2}):(\d{2})\s*(AM|PM)$/;
       for (var t = 0; t < tables.length; t++) {
         var strongs = tables[t].querySelectorAll('strong');
         for (var i = 0; i < strongs.length; i++) {
           var text = strongs[i].textContent.trim();
           var m = text.match(re);
           if (m) {
             var hours = parseInt(m[1], 10);
             var minutes = m[2];
             var ampm = m[3];
             var h24 = hours;
             if (ampm === 'AM' && h24 === 12) h24 = 0;
             if (ampm === 'PM' && h24 !== 12) h24 += 12;
             var iso = '2026-02-20T' + String(h24).padStart(2,'0') + ':' + minutes + ':00-08:00';
             strongs[i].setAttribute('data-pt-time', text);
             strongs[i].setAttribute('data-utc', iso);
             timeEls.push(strongs[i]);
           }
         }
       }
     }

     function shortTZName() {
       try {
         var parts = Intl.DateTimeFormat('en-US', {timeZone: userTZ, timeZoneName: 'short'}).formatToParts(new Date('2026-02-20T12:00:00-08:00'));
         for (var i = 0; i < parts.length; i++) {
           if (parts[i].type === 'timeZoneName') return parts[i].value;
         }
       } catch(e) {}
       return userTZ;
     }

     function toggleTimezone() {
       showingLocal = !showingLocal;
       for (var i = 0; i < timeEls.length; i++) {
         var el = timeEls[i];
         if (showingLocal) {
           try {
             var d = new Date(el.getAttribute('data-utc'));
             if (isNaN(d.getTime())) { continue; }
             var local = d.toLocaleTimeString('en-US', {timeZone: userTZ, hour: 'numeric', minute: '2-digit', hour12: true});
             el.textContent = local;
           } catch(e) {
             el.textContent = el.getAttribute('data-pt-time');
           }
         } else {
           el.textContent = el.getAttribute('data-pt-time');
         }
       }
       var btn = document.getElementById('tz-toggle-btn');
       btn.setAttribute('aria-pressed', showingLocal ? 'true' : 'false');
       if (showingLocal) {
         btn.textContent = 'Show in Pacific Time (UTC\u22128)';
       } else {
         btn.textContent = 'Show in my timezone (' + shortTZName() + ')';
       }
     }

     document.addEventListener('DOMContentLoaded', function() {
       findTimeElements();
       if (!isSameTZ && timeEls.length > 0) {
         var btn = document.getElementById('tz-toggle-btn');
         btn.textContent = 'Show in my timezone (' + shortTZName() + ')';
         btn.style.display = 'inline-block';
         btn.addEventListener('click', toggleTimezone);
       }
     });
   })();
   </script>

Morning Session
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 15 85
   :header-rows: 0

   * - **9:00 AM**
     - **Introductory Remarks**

   * - **9:30 AM**
     - | **HiGHS for CVXPY** — `Julian Hall <#speaker-julian-hall>`_ *(Virtual)*
       |
       | HiGHS is the world's best open-source linear optimization software. Although it's been callable from CVXPY for a few years, its presence has grown, and it is the default MIP solver from CVXPY 1.8. This talk introduces HiGHS: its history, its solvers, its users, its people and its future.

   * - **10:00 AM**
     - | **Adapting CVXPY for Optimization on Curved Spaces** — `Nikhesh Kumar Murali <#speaker-nikhesh-kumar-murali>`_ *(Virtual)*
       |
       | Convex optimization has emerged as a fundamental approach in machine learning, AI systems, and core engineering fields due to its global optimality guarantees and progress in computational techniques. Modern frameworks like CVXPY enable high-level problem specification via Python and offer high precision solvers. However, adapting CVXPY convex optimization solver for curved spaces present some problems like including hierarchical representation learning, hyperbolic embeddings, and structured latent space modeling in neural systems, where classical notions of optimisation and convexity on Euclidean spaces are challenged. We explore some strategies for adapting, redefining those notions for curved spaces, especially hyperbolic spaces with g-convexity and its refinement h-convexity. While g-convexity generalises convexity to curved spaces where local minima coincide with global minima under appropriate conditions, the concept of h-convexity introduces curvature independent guarantees on certain curved spaces with non-positive sectional curvature called as Hadamard manifolds. We propose some techniques to model proximal problems via tangent spaces, integrate g-convexity & h-convexity into solvers, and construct differentiable optimization layers in CVXPY to support the convex sub problems for curved spaces.

   * - **10:10 AM**
     - | **Disciplined Biconvex Programming** — `Hao Zhu <#speaker-hao-zhu>`_ *(Virtual)*
       |
       | We introduce disciplined biconvex programming (DBCP), a modeling framework for specifying and solving biconvex optimization problems. Biconvex optimization problems arise in various applications, including machine learning, signal processing, computational science, and control. Solving a biconvex optimization problem in practice usually resolves to heuristic methods based on alternate convex search (ACS), which iteratively optimizes over one block of variables while keeping the other fixed, so that the resulting subproblems are convex and can be efficiently solved. However, designing and implementing an ACS solver for a specific biconvex optimization problem usually requires significant effort from the user, which can be tedious and error-prone. DBCP extends the principles of disciplined convex programming to biconvex problems, allowing users to specify biconvex optimization problems in a natural way based on a small number of syntax rules. The resulting problem can then be automatically split and transformed into convex subproblems, for which a customized ACS solver is then generated and applied. DBCP allows users to quickly experiment with different biconvex problem formulations, without expertise in convex optimization. We implement DBCP into the open source Python package dbcp, as an extension to the famous domain specific language CVXPY for convex optimization.

   * - **10:40 AM**
     - | **Optimal Matching with CVXPY** — `Michael Howes <#speaker-michael-howes>`_ *(In-Person)*
       |
       | The Stanford Statistics Department uses CVXPY to assign PhD students to courses to TA based on their preferences. The assignments are made by gathering the students' preferences and then solving a linear program to find an optimal matching. Although the linear program is a relaxation of the original objective, CVXPY returns a valid assignment if the solver is correctly specified. CVXPY also makes it easy to add a variety of constraints based on students' experience and availability.

   * - **10:50 AM**
     - | **Panel: Contributing to CVXPY** — William Zhang & Nikhil Devenathan *(In-Person)*

   * - **11:05 AM**
     - | **Talkback: CVXPY New Features** *(Hybrid)*

   * - **11:30 AM**
     - | **Optimizing Water and Wastewater Treatment Systems with CVXPY** — `Daly Wettermark <#speaker-daly-wettermark>`_ *(In-Person)*
       |
       | Pumping and treatment of water comprises approximately 4% of global electricity consumption, with up to 20% in regions where water scarcity necessitates advanced treatment or long-distance pumping like California. Recent research has highlighted the potential for water systems to reduce their electricity-related costs and emissions through flexible operation. In this presentation, we present an overview of how water sector facilities such as desalination, water distribution, and wastewater treatment are well positioned to contribute to grid flexibility. We will specifically discuss an optimization problem in CVXPY for a wastewater treatment facility with a biogas microgrid, using physics-informed constraints and real-world electricity tariffs and Scope 2 emissions factors. We will discuss the results, which demonstrate the potential for energy flexibility to reduce operating costs by 12% compared to baseline operation, and how the optimization tool can be used to compare different flexibility technologies.

   * - **11:40 AM**
     - | **A Tutorial on Graphs of Convex Sets** — `Tobia Marcucci <#speaker-tobia-marcucci>`_ *(In-Person)*
       |
       | In this talk, I will give a tutorial on graphs of convex sets, with emphasis on their applications in robotics, planning, and, more broadly, decision making. Mathematically, a Graph of Convex Sets (GCS) is a graph in which vertices are associated with convex optimization problems and edges couple pairs of these problems through additional convex costs and constraints. Classical problems defined over ordinary weighted graphs (such as the shortest path, the traveling salesman, and the minimum spanning tree) naturally generalize to a GCS, giving rise to a rich class of problems at the interface of combinatorial and convex optimization. First, I will discuss how GCS problems can be solved efficiently and show a variety of real-world applications. Secondly, I will present GCSOPT, an open-source and easy-to-use Python library, based on CVXPY, that enables solving complex GCS problems in just a few lines of code.

   * - **12:10 PM**
     - **Lunch Break**

Afternoon Session
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 15 85
   :header-rows: 0

   * - **1:15 PM**
     - | **Disciplined Nonlinear Programming** — `Daniel Cederberg <#speaker-daniel-cederberg>`_ *(In-Person)*
       |
       | In this talk I introduce disciplined nonlinear programming (DNLP), a new syntax for specifying nonlinear programming problems. DNLP allows smooth functions to be freely mixed with nonsmooth convex and concave functions, with rules governing how the nonsmooth functions can be used. Problems expressed in DNLP form can be automatically canonicalized to a standard nonlinear programming (NLP) form and passed to a suitable NLP solver. I conclude by describing the DNLP language in more detail, comparing it with existing NLP modeling languages, and presenting our open-source implementation of DNLP as an extension of CVXPY.

   * - **1:45 PM**
     - | **pycvxset: Convex Sets in Python for Control and Analysis** — `Abraham Vinod <#speaker-abraham-vinod>`_ *(Virtual)*
       |
       | pycvxset is a new Python package to manipulate and visualize convex sets. We support polytopes and ellipsoids, and provide user-friendly methods to perform a variety of set operations. For polytopes, pycvxset supports the standard halfspace/vertex representation as well as the constrained zonotope representation. The main advantage of constrained zonotope representations over standard halfspace/vertex representations is that constrained zonotopes admit closed-form expressions for several set operations. pycvxset uses CVXPY to solve various convex programs arising in set operations, and uses pycddlib to perform vertex-halfspace enumeration. We demonstrate the use of pycvxset in analyzing and controlling dynamical systems in Python. pycvxset is available at https://github.com/merlresearch/pycvxset under the AGPL-3.0-or-later license, along with documentation and examples.

   * - **2:00 PM**
     - | **Riskfolio-Lib: Advanced Portfolio Optimization** — `Dany Cajas <#speaker-dany-cajas>`_ *(Virtual)*
       |
       | Advanced portfolio optimization with convex and integer programming in Python.

   * - **2:15 PM**
     - | **Talkback: Serialization Specifications** *(Hybrid)*

   * - **2:45 PM**
     - | **Minimizing Electricity-Related Costs and Emissions with EECO** — `Fletcher Chapin <#speaker-fletcher-chapin>`_ *(In-Person)*
       |
       | Electricity tariff structures are extremely complex due to features such as time-of-use pricing, tiered charges, and bundling (whether generation and distribution are billed together). Industrial energy flexibility, or the ability of industrial electricity consumers to modify the timing of electricity consumption, has been the subject of much research due to its potential to reduce both costs and emissions. However, the magnitudes of bill savings or carbon emission reductions are a strong function of the prevailing electricity tariffs. In this work, we expand on our previous research, which compiled industrial electricity tariffs from across the United States, by presenting Electric Emissions & Cost Optimizer (EECO), a software package for optimizing electricity-related emissions and costs using the format from our published tariff dataset. After briefly discussing the technical details of EECO, we present a simple case study demonstrating how to use EECO to optimize an industrial consumer's electricity bill with a simple CVXPY battery model. The case study emphasizes the ease with which users can select from thousands of available tariffs representing the entire United States.

   * - **3:00 PM**
     - | **Development Plans for CVXR** — `Balasubramanian Narasimhan <#speaker-balasubramanian-narasimhan>`_ & Anqi Fu *(In-Person)*
       |
       | CVXR is the R port of CVXPY and has a developer community of two. As a result, it has fallen behind CVXPY in features but new AI development tools plus facilities in R offer hope of catching up. In this talk we describe our plans and present data from some early experiments that sound promising. The talk will be quite interactive and we welcome feedback from seasoned CVXPY community.

   * - **3:15 PM**
     - | **Moreau: The Convex Optimization Solver for the Modern Workload** — `Steven Diamond <#speaker-steven-diamond>`_ *(In-Person, Sponsored)*

   * - **3:30 PM**
     - | **Electrified Transportation Optimization with CVXPY** — `Kyle Goodrick <#speaker-kyle-goodrick>`_ *(In-Person)*
       |
       | To achieve the lowest cost operation, electrified transportation must maximize the utilization of charging and grid infrastructure. This presentation introduces a site optimization tool designed to identify the lowest-cost electrification strategy for shared transportation infrastructure serving multiple sectors, including heavy-duty trucks, buses, regional rail, and light rail. Implemented with CVXPY, the tool models the system as a convex optimization problem that jointly considers energy demand profiles, infrastructure constraints, and cost drivers across the site and the upstream power system. The model evaluates tradeoffs between on-site investments and grid upgrades, considering interconnection capacity limits, distribution impacts, and utility rate structures. It optimizes on-site energy storage and co-optimizes across transportation modes to reduce total system cost. This presentation will describe a real-world application of CVXPY where coordinated planning across transportation sectors can minimize total cost, reduce emissions, and more fully utilize existing grid infrastructure. The tool is intended to support fleets, utilities, planners, and policymakers in evaluating scalable, economically efficient approaches to transportation electrification at complex, multi-user sites.

   * - **3:45 PM**
     - | **MOCVXPY: A CVXPY Extension for Multiobjective Optimization** — `Ludovic Salomon <#speaker-ludovic-salomon>`_ *(Virtual)*
       |
       | MOCVXPY is a library built on top of CVXPY for convex vector optimization. It enables practitioners to describe their convex vector optimization problems using an intuitive algebraic language that closely follows the mathematical formulation. This talk presents the main features of MOCVXPY and the algorithms employed by the library. We also illustrate its functionality with examples and an application in energy.

   * - **4:00 PM**
     - | **Sponsored Talk** *(TBD)*

   * - **4:15 PM**
     - | **Keynote** — `Stephen Boyd <#speaker-stephen-boyd>`_ *(In-Person)*

   * - **5:00 PM**
     - | **Social** — `Accel Partners, Palo Alto <https://maps.app.goo.gl/akDpcVErYi859q6V9>`_

Hackathon (February 21)
-----------------------

Details coming soon.

Speakers
--------

.. dropdown:: Julian Hall
   :name: speaker-julian-hall

   Julian Hall has a BA in Mathematics from the University of Oxford, a PhD from the University of Dundee supervised by Roger Fletcher (FRS) and, since 1990, has been employed as a lecturer by the University of Edinburgh. His occasional research activity has been in the field of computational techniques for continuous linear optimization on serial and parallel architectures, notably the simplex method. His work (with others) has yielded journal articles that have won four best paper prizes. Since the mid-90s, he has been engaged in consultancy, notably providing linear optimization technology to Format International, whose software formulates half the world's manufactured animal feed. In 2016, with Ivet Galabova, he founded HiGHS, which has grown to be the world's best open-source linear optimization software.

.. dropdown:: Hao Zhu
   :name: speaker-hao-zhu

   Hao Zhu completed his bachelor's degree in computational chemistry at Nankai University in China, followed by a master's degree in neuroscience at the University of Freiburg, Germany. He is currently a PhD student in the Department of Computer Science at the University of Freiburg, jointly supervised by Prof. Joschka Boedecker and Prof. Moritz Diehl.

.. dropdown:: Michael Howes
   :name: speaker-michael-howes

   Michael Howes is a fifth year graduate student in the Statistics Department at Stanford University. He works on the theory of auxiliary variables Markov Chain Monte Carlo algorithms and is advised by Prof. Persi Diaconis. He has also published interdisciplinary work on using large language models for valid inference in computational social science. In Summer 2025, Michael was a data science intern at Waymo where he designed algorithms for improved triage sampling.

.. dropdown:: Nikhesh Kumar Murali
   :name: speaker-nikhesh-kumar-murali

   Nikhesh Kumar Murali is a Data Scientist and interdisciplinary researcher working at the intersection of mathematics, optimization, and modern artificial intelligence. He holds an undergraduate degree in Mathematics from the Indian Institute of Science, Bengaluru, and a joint Master's degree in Artificial Intelligence and Data Science from IIT Madras and the University of Birmingham. His research interests span convex optimization, AI4Mathematics, and geometry of latent spaces for mechanistic interpretability. He has worked on topics ranging from metric geometry and Fuchsian groups, automated theorem proving using Lean to Responsible AI and computer vision tasks such as object detection. He is particularly focused on how tools from differential/hyperbolic geometry, optimal transport, and convex analysis can be integrated into solving fundamental problems. Currently, Nikhesh works as a Data Scientist in industry, where he designs and deploys machine learning systems and Agentic AI architectures for code automation.

.. dropdown:: Tobia Marcucci
   :name: speaker-tobia-marcucci

   Tobia Marcucci is an Assistant Professor in the Department of Electrical and Computer Engineering at the University of California, Santa Barbara (UCSB). He is also affiliated with the Department of Mechanical Engineering and the Center for Control, Dynamical Systems, and Computation (CCDC) at UCSB. Before joining UCSB, he was a Postdoctoral Scientist at Amazon Robotics. He received a PhD in Computer Science from the Massachusetts Institute of Technology (MIT), under the supervision of Russ Tedrake and Pablo Parrilo. During his PhD, he also spent one year at Stanford University visiting Stephen Boyd's group. His doctoral dissertation was awarded the 2025 MIT EECS George M. Sprowls Thesis Award in Artificial Intelligence and Decision Making. He holds a Bachelor's and a Master's degree cum laude in Mechanical Engineering from the University of Pisa. His research lies at the intersection of convex and combinatorial optimization, with applications to robotics, motion planning, and optimal control.

.. dropdown:: Daniel Cederberg
   :name: speaker-daniel-cederberg

   Daniel Cederberg is a PhD student at Stanford University, advised by Stephen Boyd.

.. dropdown:: Abraham Vinod
   :name: speaker-abraham-vinod

   Abraham P. Vinod is a Principal Research Scientist at Mitsubishi Electric Research Laboratories (MERL), Cambridge, MA, USA. His main research interests are in the areas of constrained control, robotics, and learning. Prior to joining MERL in 2020, he held a postdoctoral position at the University of Texas at Austin. He received his B.Tech. and M.Tech. degrees from the Indian Institute of Technology-Madras (IIT-M), India, and his Ph.D. degree from the University of New Mexico, USA, all in electrical engineering. He was the recipient of the Best Student Paper Award at the 2017 ACM Hybrid Systems: Computation and Control Conference, Finalist for the Best Paper Award in the 2018 ACM Hybrid Systems: Computation and Control Conference, and the best undergraduate student research project award at IIT-M. He is the primary developer of pycvxset, a set computation toolbox in Python, and co-developed SReachTools, a stochastic reachability toolbox in MATLAB.

.. dropdown:: Daly Wettermark
   :name: speaker-daly-wettermark

   Daly Wettermark is a PhD student in the WE3 Lab in Civil and Environmental Engineering. She studies the potential to mitigate greenhouse emissions from wastewater treatment plants by optimizing novel resource recovery and energy management techniques. Her PhD focus areas include energy flexibility as a means to shift load from aeration and comparison of nitrogen recovery mechanisms for reducing biological nutrient removal process intensity. Daly previously worked as a product development engineer and corporate sustainability analyst and as a volunteer with sanitation-focused nonprofit organizations.

.. dropdown:: Dany Cajas
   :name: speaker-dany-cajas

   Dany Cajas holds a BSc in Economic Engineering and Master in Finance. He is the creator and sole maintainer of Riskfolio-Lib (3,723 Github stars and +1.1M downloads) and author of "Advanced Portfolio Optimization: a Cutting-edge Quantitative Approach".

.. dropdown:: Fletcher Chapin
   :name: speaker-fletcher-chapin

   Fletcher Chapin is a 5th-year environmental engineering PhD candidate advised by Dr. Meagan Mauter in the WE3Lab. His research focuses on data management solutions for the water sector, minimizing costs and emissions of wastewater treatment through flexible operation, and formal verification of proposed control schemes. Before coming to Stanford, Fletcher graduated from Cornell with a BS in Environmental Engineering and minor in Computer Science. After Cornell, he first worked as a software engineer at Microsoft before returning to the water sector as India Program Project Manager for AguaClara Reach. Fletcher will be defending his dissertation in May, so we'll see what comes next!

.. dropdown:: Balasubramanian Narasimhan
   :name: speaker-balasubramanian-narasimhan

   Balasubramanian Narasimhan is a Senior Research Scientist in Departments of Biomedical Data Sciences and Statistics at Stanford. Together with Anqi Fu, he is the maintainer of CVXR, the R port of CVXPY.

.. dropdown:: Kyle Goodrick
   :name: speaker-kyle-goodrick

   Kyle Goodrick is a Senior Power Systems Research Engineer at ASPIRE, where he focuses on research at the intersection of electric power systems and electrified transportation. His work is centered on enabling economic growth while reducing costs and emissions through better utilization of existing grid infrastructure. He is particularly interested in how large, flexible electric loads, primarily electric vehicles, can be coordinated with the power system to improve efficiency, defer infrastructure upgrades, and support reliable system operation. Kyle earned his PhD in Electrical Engineering from the University of Colorado Boulder, studying the optimization of power distribution architectures.

.. dropdown:: Steven Diamond
   :name: speaker-steven-diamond

   Steven Diamond is the founder of CVXPY and its head maintainer. He is also a founder and COO of Optimal Intellect.

.. dropdown:: Ludovic Salomon
   :name: speaker-ludovic-salomon

   Ludovic Salomon is a researcher at Hydro-Québec, the primary producer and distributor of electricity in Quebec, Canada. He holds a Ph.D. in applied mathematics and computer science from Polytechnique Montréal. His research interests are in numerical optimization, at the intersection of computer science and mathematics, with a focus on multiobjective optimization.

.. dropdown:: Stephen Boyd
   :name: speaker-stephen-boyd

   Stephen P. Boyd is the Samsung Professor of Engineering, Professor of Electrical Engineering, and a member of the Institute for Computational and Mathematical Engineering at Stanford University. His current research focus is on convex optimization applications in control, signal processing, machine learning, and finance. He has developed and taught many undergraduate and graduate courses, including Signals & Systems, Linear Dynamical Systems, Convex Optimization, and a recent undergraduate course on Matrix Methods. His graduate convex optimization course attracts 300 students from 25 departments. In 2003, he received the AACC Ragazzini Education award, for contributions to control education. In 2016 he received the Walter J. Gores award, the highest award for teaching at Stanford University. In 2017 he received the IEEE James H. Mulligan, Jr. Education Medal, the highest education award of the IEEE, for a career of outstanding contributions to education.

Contact
-------

For questions about the workshop, please email cvxpydevs@gmail.com.
