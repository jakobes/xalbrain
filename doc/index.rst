.. cbcbeat documentation master file, created by
   sphinx-quickstart on Fri Sep 12 10:27:56 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to cbcbeat's documentation!
===================================

cbcbeat is a Python-based lightweight solver collection for solving
cardiac electrophysiology problems. cbcbeat efficiently solves single
cardiac cell models, monodomain and bidomain, both forward and inverse
problems.

cbcbeat originates from the `Center for Biomedical Computing
<http://cbc.simula.no>`__, a Norwegian Centre of Excellence, hosted by
`Simula Research Laboratory <http://www.simula.no>`__, Oslo, Norway.


Installation and dependencies:
------------------------------

The cbcbeat source code is hosted on Bitbucket:

  https://bitbucket.org/meg/cbcbeat

cbcbeat is based on

* The FEniCS Project software (http://www.fenicsproject.org)
* dolfin-adjoint (http://www.dolfin-adjoint.org)

See the separate file ./INSTALL in the root directory of your cbcbeat
source for a complete list of dependencies.

Main authors:
-------------

  * Marie E. Rognes    (meg@simula.no)
  * Johan E. Hake      (hake@simula.no)
  * Patrick E. Farrell (patrick.farrell@maths.ox.ac.uk)
  * Simon W. Funke     (simon@simula.no)

API documentation:
------------------

.. toctree::
   :maxdepth: 2

   demo/*
   cbcbeat


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
