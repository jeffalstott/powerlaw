from setuptools import setup
with open('README.rst') as file:
        long_description = file.read()

### Crap to be able to create a binary installer for Windows
import codecs
try:
    codecs.lookup('mbcs')
except LookupError:
    ascii = codecs.lookup('ascii')
    func = lambda name, enc=ascii: {True: enc}.get(name=='mbcs')
    codecs.register(func)

setup(
    name='powerlaw',
    py_modules=['powerlaw'],
    version= '1.4.6',
    description='Toolbox for testing if a probability distribution fits a power law',
    long_description=long_description,
    author='Jeff Alstott',
    author_email='jeffalstott@gmail.com',
    url='http://www.github.com/jeffalstott/powerlaw',
    install_requires=['scipy', 'numpy', 'matplotlib', 'mpmath'],
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research'
    ]
)
