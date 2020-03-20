Bootstrap: docker
From: quay.io/fenicsproject/stable:2019.1.0.r3

%files
    install_fenicstools.sh ${FENICS_HOME}

%post
    sudo apt-get install libmpich-dev libhdf5-mpich-dev mpich
    python3 -m pip install --upgrade pip
    python3 -m pip install tqdm h5py cppimport
    # python3 -m pip install git+git://github.com/jakobes/xalbrain@js-2020
    python3 -m pip install git+git://github.com/jakobes/xalpost
    python3 -m pip install git+git://github.com/jakobes/xalode
    python3 -m pip install --user git+git://github.com/jakobes/fenicstools

    cd ${FENICS_HOME}
    git clone https://github.com/jakobes/fenicstools.git
    ldconfig

%runscript
    exec /bin/bash -i
