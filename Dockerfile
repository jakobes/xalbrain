# Builds a Docker image for reproducing the results in the wetting and drying
# adjoint paper by Funke et.al

FROM quay.io/dolfinadjoint/dev-dolfin-adjoint:latest
MAINTAINER Simon Funke <simon@simula.no>

USER root
RUN sudo apt-get update && sudo apt-get -y install mercurial

USER fenics
RUN hg clone https://bitbucket.org/meg/cbcbeat
RUN cd cbcbeat && pip install . --user

USER root
