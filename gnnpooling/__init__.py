"""
gnnpooling
Genetic algorithm for molecule optimisation
"""

# Add imports here
from gnnpooling.runner import run_experiment

# Handle versioneer
from gnnpooling._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
