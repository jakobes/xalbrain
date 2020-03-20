#/usr/bin/env bash

git clone https://github.com/jakobes/fenicstools.git
cd fenicstools
python3 -m pip install --user .
python3 -c "import fenicstools; fenicstools.Probe"
