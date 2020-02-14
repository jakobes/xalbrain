#/usr/bin/env bash

cd ${FENICS_HOME}/fenicstools
${FENICS_PYTHON} -m pip install --user .
${FENICS_PYTHON} -c "import fenicstools; fenicstools.Probe"
