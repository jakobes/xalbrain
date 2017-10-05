from fenics import *
import instant
import os
import time


def _get_cpp_module(source_dir, header_files, source_files, force_recompile=False):
    """Use the dolfin machinery to compile, wrap with swig and load a c++ module.

    # modified from ocellaris
    https://bitbucket.org/trlandet/ocellaris
    """

    cpp_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(cpp_dir, source_dir)

    header_sources = []
    for hpp_filename in header_files:
        hpp_filename = os.path.join(source_dir, hpp_filename)
        
        with open(hpp_filename, 'rt') as f:
            hpp_code = f.read()
        header_sources.append(hpp_code)

    # Force recompilation
    if force_recompile:
        header_sources.append('// %s \n' % time.time())

    # def _readSourceFile(filepath):
    #     with open(os.path.join(source_dir, filepath), "r") as _source_file:
    #         return _source_file.read()
    #
    #
    # code = "\n\n\n".join(map(_readSourceFile, source_files))
    
    try:
        module = compile_extension_module(
            code='\n\n\n'.join(header_sources),
            source_directory=source_dir, 
            sources=source_files,
            include_dirs=[".", source_dir]
        )

        # module = compile_extension_module(
        #     code,       # Header
        #     additional_system_headers=additional_headers
        # )
    except RuntimeError as e:
        try:
            with open(e.__str__().split("'")[-2]) as errfile:
                print(errfile.read())
        except IndexError as ie:
            print(ie)
        raise

    return module


class _ModuleCache(object):
    def __init__(self):
        """
        A registry and cache of available C/C++ extension modules
        """
        self.available_modules = {}
        self.module_cache = {}
    
    def add_module(self, name, source_dir, header_files, source_files):
        """
        Add a module that can be compiled
        """
        self.available_modules[name] = (source_dir, header_files, source_files)
        
    def get_module(self, name, force_recompile=False):
        """
        Compile and load a module (first time) or use from cache (subsequent requests)
        """
        if force_recompile or name not in self.module_cache:
            source_dir, header_files, source_files = self.available_modules[name]
            mod = _get_cpp_module(source_dir, header_files, source_files, force_recompile)
            self.module_cache[name] = mod
        
        return self.module_cache[name]


_MODULES = _ModuleCache()
_MODULES.add_module(
    "AdexPointIntegralSolver", "adex", 
    [
        "AdexPointIntegralSolver.h",
    ], [
        "AdexPointIntegralSolver.cpp",
    ]
)


def load_module(name, force_recompile=False):
    """Load the C/C++ module registered with the given name.

    Reload forces a cache-refresh, otherwise subsequent accesses are cached
    """
    return _MODULES.get_module(name, force_recompile)
