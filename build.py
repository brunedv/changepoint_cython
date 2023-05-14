import os
import shutil
from distutils.core import Distribution
from distutils.extension import Extension

from Cython.Build import build_ext, cythonize



extensions = [Extension('pychangepoints.cython_pelt', ['pychangepoints/pelt_cython.pyx', 'pychangepoints/cost_function.pxd', 'pychangepoints/utils.pxd'], extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"], extra_link_args=['-fopenmp']),
              Extension('pychangepoints.multiple_dim', ['pychangepoints/mutiple_algos.pyx', 'pychangepoints/cost_function_multiple.pxd']),
              Extension('pychangepoints.nonparametric', ['pychangepoints/nppelt.pyx'])]

ext_modules = cythonize(extensions, compiler_directives={'language_level': 3})
dist = Distribution({"ext_modules": ext_modules})
cmd = build_ext(dist)
cmd.ensure_finalized()
cmd.run()