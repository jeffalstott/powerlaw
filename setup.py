from distutils.core import setup
setup(
        name = 'powerlaw',
        py_modules = ['powerlaw'],
        version = '.4.2',
        description = 'Toolbox for testing if a probability distribution fits a power law',
        author='Jeff Alstott',
        author_email = 'jeffalstott@gmail.com',
        url = 'http://code.google.com/p/powerlaw/',
        requires = ['scipy', 'numpy', 'matplotlib', 'mpmath'],
        license = 'MIT',
        classifiers = [
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2.7',
            'Operating System :: OS Independent',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research'
            ]
        )
