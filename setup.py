"""
Setup file
"""
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize, build_ext
import numpy

with open("README.md", "r") as fh:
    long_description = fh.read()

extensions = [Extension('pychangepoints.cython_pelt', ['pychangepoints/pelt_cython.pyx', 'pychangepoints/cost_function.pxd', 'pychangepoints/utils.pxd'], extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"], extra_link_args=['-fopenmp']),
              Extension('pychangepoints.multiple_dim', ['pychangepoints/mutiple_algos.pyx', 'pychangepoints/cost_function_multiple.pxd']),
              Extension('pychangepoints.nonparametric', ['pychangepoints/nppelt.pyx'])]
setup(
    name='changepoint_cython',
    version='0.1.1',
    author='Vianney Bruned',
    install_requires=['Cython', 'numpy', 'pandas', 'scikit-learn'],
    author_email="vianney.bruned@gmail.com",
    description="A cython version of the changepoint R package",
    url="https://https://github.com/brunedv/changepoint_cython",
    packages=["pychangepoints"],
    include_package_data=True,
    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': build_ext},
    include_dirs=[numpy.get_include()]
)
