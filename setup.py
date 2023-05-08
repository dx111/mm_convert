from setuptools import setup
from setuptools import find_packages

setup(
    name='mm_convert',
    version='0.0.3',
    packages=find_packages(),
    zip_safe = False,
    entry_points={
        'console_scripts': [
            'mm_convert = mm_convert:main',
        ]
    },
    package_data={'':['sample_data/*']}
)