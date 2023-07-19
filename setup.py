from setuptools import setup, find_packages

setup(name='mygravipy',
      version='1.3',
      author='Felix Widmann',
      description='Package to work with GRAVITY GC data',
      url='https://github.com/widmannf/mygravipy',
      python_requires='>=3.7',
      packages=['mygravipy'],
      package_dir={'':'src'},
      package_data={'gravipy': ['Phasemaps/*.npy',
                                'Datafiles/*',
                                'met_corrections/*.npy']},
      include_package_data=True,
      install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'pandas',
        'astropy',
        'emcee',
        'corner',
        'mpmath',
        'joblib',
        'lmfit',
        'numba',
        'p2api',
    ]
)
