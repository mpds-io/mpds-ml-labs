
from setuptools import setup

setup(
    name='mpds_ml_labs',
    version='0.0.7',
    author='Evgeny Blokhin',
    author_email='eb@tilde.pro',
    license='LGPL-2.1',
    packages=['mpds_ml_labs'],
    install_requires=[
        'mpds_client', 'pycodcif', 'spglib', 'sklearn', 'imblearn', 'progressbar', 'pg8000'
    ],
    python_requires='>=3.5'
)
