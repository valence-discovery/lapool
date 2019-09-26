"""
Pooling for molecular graph
"""
from setuptools import setup
import glob

short_description = __doc__.split("\n")

setup(
    # Self-descriptive entries which should always be present
    name='gnnpooling',
    description=short_description[0],
    long_description_content_type="text/markdown",
    version="alpha",

    # Which Python importable modules should be included when your package is installed
    packages=['gnnpooling'],

    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # url='http://www.my_package.com',  # Website
    # install_requires=[],              # Required packages, pulls from pip if needed; do not use for Conda deployment
    # platforms=['Linux',
    #            'Mac OS-X',
    #            'Unix',
    #            'Windows'],            # Valid platforms your code works on, adjust to your flavor
    python_requires=">=3.5",  # Python version restrictions

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,

)
