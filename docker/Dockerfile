FROM quay.io/fenicsproject/stable:2019.1.0.r3

USER root

ENV FENICS_PYTHON=python3

RUN ${FENICS_PYTHON} -m pip install --upgrade pip && \
    ${FENICS_PYTHON} -m pip install tqdm h5py cppimport && \
    ${FENICS_PYTHON} -m pip install git+git://github.com/jakobes/xalbrain@js-2018 && \
    ${FENICS_PYTHON} -m pip install git+git://github.com/jakobes/xalpost && \
    ${FENICS_PYTHON} -m pip install git+git://github.com/jakobes/xalode

RUN cd ${FENICS_HOME} && \
    git clone https://github.com/jakobes/fenicstools.git

COPY install.sh $HOME
