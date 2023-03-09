import re
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize, build_ext
import numpy

extensions = [
    Extension(
        "agox.models.GPR.priors.repulsive",
        ["agox/models/GPR/priors/repulsive.pyx"],
        include_dirs=[numpy.get_include()]
        ),
    Extension(
        "agox.models.descriptors.fingerprint_cython.angular_fingerprintFeature_cy",
        ["agox/models/descriptors/fingerprint_cython/angular_fingerprintFeature_cy.pyx"],
        include_dirs=[numpy.get_include()]
        ),
    ]
    
# Version Number:
version_file = 'agox/__version__.py'
with open(version_file) as f:
    lines = f.readlines()

for line in lines:
    if '__version_info__' in line:
        result = re.findall('\d+', line)
        result = [int(x) for x in result]
        version = '{}.{}.{}'.format(*result)
        break

setup(
    name="agox",
    version=version,
    url="https://agox.gitlab.io/agox/",
    description="Atomistic Global Optimziation X is a framework for structure optimization in materials science.",
    install_requires=[
        "numpy >=1.18,<1.24",
        "ase",
        "matplotlib",
        "cymem",
        "scikit-learn",
    ],
    extras_require={
        'full': [
            "dscribe",
            "ray==2.0.0",
            "pytest",
            "jax",
        ]
    },
    python_requires=">=3.5",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=cythonize(extensions),
    entry_points={'console_scripts':['agox-convert=agox.utils.convert_database:convert', 
                                     'agox-analysis=agox.utils.batch_analysis:command_line_analysis']})
