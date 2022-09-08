from importlib.metadata import entry_points
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
    Extension(
        "agox.modules.models.priors.repulsive",
        ["agox/modules/models/priors/repulsive.pyx"],
        include_dirs=[numpy.get_include()]
    ),        
]

setup(
    name="agox",
    version="1.2.0",
    url="https://gitlab.com/agox/agox",
    description="Atomistic Global Optimziation X is a framework structure optimization in materials science.",
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
    entry_points={'console_scripts':['agox-convert=agox.utils.convert_database:convert', 
                                     'agox-analysis=agox.utils.batch_analysis:command_line_analysis']})
