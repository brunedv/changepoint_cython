from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize, build_ext
#import Cython.Compiler.Options
#Cython.Compiler.Options.annotate = True
import numpy

extensions = [Extension('pychangepoints.cython_pelt',['pychangepoints/pelt_cython.pyx','pychangepoints/cost_function.pxd'
                                                       'pychangepoints/utils.pxd']),
             Extension('pychangepoints.multiple_dim',['pychangepoints/mutiple_algos.pyx','pychangepoints/cost_function_multiple.pxd'])]
setup(
    name = 'changepoint_cython',
    version = '0.1',
    author = 'Vianney Bruned',
    install_requires = ['Cython','numpy','pandas'],
    packages = ["pychangepoints"],
    include_package_data=True,
    ext_modules = cythonize( extensions ),
    cmdclass = { 'build_ext' : build_ext },
    include_dirs=[numpy.get_include()] ,
)