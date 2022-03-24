from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "agox.modules.models.gaussian_process.featureCalculators_multi.angular_fingerprintFeature_cy",
        ["agox/modules/models/gaussian_process/featureCalculators_multi/angular_fingerprintFeature_cy.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "agox.modules.models.gaussian_process.delta_functions_multi.delta",
        ["agox/modules/models/gaussian_process/delta_functions_multi/delta.pyx"],
        include_dirs=[numpy.get_include()]
    ),    
]

setup(
    name="agox",
    version="0.1.0",
    url="agox.au.dk...",
    description="MPs pet project of a global atomistic optimizer",
    install_requires=[
        "numpy>=1.22.0",
        "ase",
        "matplotlib",
        "cymem",
        "scikit-learn",
        "dscribe",
        "mpi4py",
    ],
    packages=find_packages(),
    python_requires=">=3.5",
    ext_modules=cythonize(extensions),
)
