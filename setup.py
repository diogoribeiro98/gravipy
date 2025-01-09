from setuptools import setup, find_packages

setup(name='gravipy',
      version='1.5',
      author='Felix Widmann',
      description='Package to work with GRAVITY GC data',
      url='https://github.com/widmannf/mygravipy',
      python_requires='>=3.7',
      packages=['gravipy'],
      package_dir={'':'src'},
      package_data={'gravipy': ['logger/config.json']},
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
        'reportlab',
        'svglib'
    ]
)
