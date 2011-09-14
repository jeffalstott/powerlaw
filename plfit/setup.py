#from distutils.core import setup
#from distutils.extension import Extension
from numpy.distutils.core import setup
from numpy.distutils.core import Extension
#from numpy.distutils.core import build_ext
from Cython.Distutils import build_ext
import Cython
import numpy

print "To create cplfit.so (for importing), call command: "
print "python setup.py build_ext --inplace"
print "If this fails, make sure c_numpy.pxd is in the path somewhere (e.g. this directory)"

try:
    from numpy.distutils.misc_util import get_numpy_include_dirs
    numpy_include_dirs = get_numpy_include_dirs()
except AttributeError:
    numpy_include_dirs = numpy.get_include()


dirs = list(numpy_include_dirs)
dirs.extend(Cython.__path__)
dirs.append('.')

ext_cplfit = Extension(
		"cplfit", 
		["cplfit.pyx"], 
		include_dirs = dirs, 
		extra_compile_args=['-O3'])

#ext_fplfit = Extension(name="fplfit",
#                    sources=["fplfit.f"])

if __name__=="__main__":
    setup(
        name = "plfit",
        version = "1.0",
        description = "Python implementation of Aaron Clauset's power-law distribution fitter",
        author = "Adam Ginsburg",
        author_email = "adam.ginsburg@colorado.edu",
        url="http://code.google.com/p/agpy/wiki/PowerLaw",
        download_url="http://code.google.com/p/agpy/source/browse/#svn/trunk/plfit",
        license = "MIT",
        platforms = ["Linux","MacOS X"],
        packages = ['plfit'],
        package_dir={'plfit':'lib'},
        install_requires = ["numpy","cython"],
        ext_modules = [ ext_cplfit ],
        cmdclass = {'build_ext': build_ext}
    )

print "I can't get numpy.distutils to compile the fortran.  To do it yourself, run some variant of:"
print 'f2py -c fplfit.f -m fplfit'

# try:
#     os.system('f2py -c fplfit.f -m fplfit')
# except:
#     print "Could not build fplfit"

