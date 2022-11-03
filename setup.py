from setuptools import setup, find_packages

setup(name="gravipy",
      version="1.1",
      author="Felix Widmann",
      description='Package to work with GRAVITY GC data',
      url="https://github.com/widmannf/gravipy",
      python_requires=">=3.7",
      packages=['gravipy'],
      package_dir={'':'src'},
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
        'fpdf',
        'numba',
        'p2api',
        'selenium',
    ]
)
