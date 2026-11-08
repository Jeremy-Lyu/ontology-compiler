"""Ontology compiler services.

This package remains the canonical flat compatibility surface while migration
subpackages provide clearer responsibility boundaries.
"""

from .extraction import *
from .quality_assurance import *
from .merge import *
from .observability import *
