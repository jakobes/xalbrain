# System imports
from distutils.core import setup
import platform
import sys
from os.path import join as pjoin

# Version number
major = 0
minor = 1


setup(
    name = "xalbrain",
    version = "{0}.{1}".format(major, minor),
    description = """An adjointable bi-domain equation solver""",
    author = "Jakob E. Schreiner",
    author_email = "jakob@xal.no",
    packages = ["xalbrain", "xalbrain.cellmodels",],
    package_dir = {"xalbrain": "xalbrain"}
)
