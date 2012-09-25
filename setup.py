# System imports
from distutils.core import setup

# Version number
major = 0
minor = 1

setup(name = "beatadjoint",
      version = "{0}.{1}".format(major, minor),
      description = """
      An adjointable bi-domain equation solver
      """,
      author = "Marie Rognes, Johan Hake",
      author_email = "meg@simula.no",
      packages = ["beatadjoint",],
      package_dir = {"beatadjoint": "beatadjoint"},
      )
