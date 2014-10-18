.. _install:

Install Guide
=============

Mac OS X
--------

1. Install the Command Line Tools for Xcode.

   Download from the `Apple developer site <http://developer.apple.com/downloads>`_.

2. Install `Anaconda`_.

   Follow the instructions on the `website`_.

3. Make sure `Anaconda`_ has the latest version of Python 2.

   ::

       conda update python


4. Install ``numpy`` and ``scipy`` using conda from the command-line.

   ::

       conda install numpy scipy

5. Install ``cvxpy`` with ``pip`` from the command-line.

   ::

       pip install cvxpy

6. Install ``nose`` using conda from the command-line.

  ::

       conda install nose

7. Test the installation with ``nose``.

  ::

       nosetests cvxpy

Ubuntu 14.04
------------

1. Make sure ``apt-get`` is up-to-date.

  ::

      sudo apt-get update

2. Install ``ATLAS`` and ``gfortran`` (needed for ``scs``).

   ::

       sudo apt-get install libatlas-base-dev gfortran

3. Install ``python-dev``.

   ::

       sudo apt-get install python-dev

4. Install ``pip``.

   ::

       sudo apt-get install python-pip

5. Install ``numpy`` and ``scipy``.

   ::

       sudo apt-get install python-numpy python-scipy

6. Install ``cvxpy``.

   ::

       sudo pip install cvxpy

6. Install ``nose``.

  ::

       sudo apt-get install python-nose

7. Test the installation with ``nose``.

  ::

       nosetests cvxpy

Windows
-------

Here is a step-by-step guide to installing cvxpy on a Windows machine.

1. If you have Python installed already, it's probably a good idea to remove it first. (Sorry!)

2. Download the latest version of Python(x,y).

3. Install Python(x,y). When prompted to select optional components, make sure to check cvxopt and MinGW, as shown below.

  .. image:: files/windows1.png
      :scale: 100%

  .. image:: files/windows2.png
      :scale: 100%


4. We need to set the default compiler as mingw32. Open Notepad and type the following, save the file at C:\Python27\Lib\distutils\distutils.cfg. (This is the default location. If you installed Python somewhere else, use the appropriate location.)

  .. image:: files/windows3.png
      :scale: 100%

5. Open Python(x,y) and launch the interactive console (highlighted button in the picture). This will bring up a console.

  .. image:: files/windows4.png
      :scale: 100%

6. From the console, run "pip install ecos" to install ecos.

7. We need to install BLAS and LAPACK libraries, and make the scs package use them. Go here to download the win32 version of the dll and lib files of both BLAS and LAPACK. Put them under some directory, say C:\blaslapack, as shown below.

  .. image:: files/windows5.png
      :scale: 100%

8. The system needs to know where to find the libraries. Right click on This PC (or My Computer), click Properties, Advanced system settings, then Environment Variables. Under the System variables list, find a variable named Path, and press Edit. Then, at the end of the list, put the address to the directory where you put the library files. All paths must be separated by semicolons.

  .. image:: files/windows6.png
      :scale: 100%

9. Go here and download the scs package as a zip file. Unzip it.

10. Browse to scs-master directory, and edit line 48 of the file scs.mk to "USE_LAPACK = 1". Without this, scs won't be able to solve SDPs.

  .. image:: files/windows7.png
      :scale: 100%

11. Browse to the src directory, and open the file cones.c. Edit lines 11 and 13 to look like the following.

  .. image:: files/windows8.png
      :scale: 100%

12. We have to change the numpy settings so that it knows where to find the libraries. Open C:\Python27\Lib\site-packages\numpy\distutils\site.cfg and add the following lines to the end of the file:

  ::

    [blas]
    library_dirs = C:\blaslapack
    blas_libs = blas
    [lapack]
    library_dirs = C:\blaslapack
    lapack_libs = lapack

You can remove what's already in there, and replace the file with just the six lines above.

  .. image:: files/windows9.png
      :scale: 100%

13. Go back to the Python(x,y) terminal, and browse to the python directory of scs-master. From there, type "python setup.py build" to build scs. (If this step results in some error, remove the build directory and try again.) After the build is successful, run "python setup.py install" to install.

14. After scs is installed, run "pip install cvxpy" to install cvxpy.

15. Reboot your computer so that the path environment variable we set in step 8 takes effect.

16. cvxpy should work now. You can use the Spyder IDE from the Python(x,y) home window. Click on the Spyder button to launch it. This IDE allows you to code, run, and view the console all in the same window. In order to check if the installation was successful, open a terminal, browse to C:\Python27\Lib\site-packages\cvxpy, and run "nosetests tests". This runs all unit tests and reports any error found.

  .. image:: files/windows10.png
      :scale: 50%

Other Platforms
---------------

The CVXPY installation process on other platforms is less automated and less well tested. Check `this page <https://github.com/cvxgrp/cvxpy/wiki/CVXPY-installation-instructions>`_ for instructions for your platform.

Install from source
-------------------

CVXPY has the following dependencies:

* Python 2.7
* `setuptools`_ >= 1.4
* `toolz`_
* `CVXOPT`_ >= 1.1.6
* `ECOS`_ >= 1.0.3
* `SCS`_ >= 1.0.1
* `NumPy`_ >= 1.8
* `SciPy`_ >= 0.13

To test the CVXPY installation, you additionally need `Nose`_.

CVXPY automatically installs `ECOS`_, `CVXOPT`_, `SCS`_, and `toolz`_.
`NumPy`_ and `SciPy`_ will need to be installed manually. Once you’ve
installed `NumPy`_ and `SciPy`_, installing CVXPY from source is simple:

1. Clone the `CVXPY git repository`_.
2. Navigate to the top-level of the cloned directory and run

   ::

       python setup.py install

.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _website: https://store.continuum.io/cshop/anaconda/
.. _setuptools: https://pypi.python.org/pypi/setuptools
.. _toolz: http://github.com/pytoolz/toolz/
.. _CVXOPT: http://cvxopt.org/
.. _ECOS: http://github.com/ifa-ethz/ecos
.. _SCS: http://github.com/cvxgrp/scs
.. _NumPy: http://www.numpy.org/
.. _SciPy: http://www.scipy.org/
.. _Nose: http://nose.readthedocs.org
.. _CVXPY git repository: https://github.com/cvxgrp/cvxpy
