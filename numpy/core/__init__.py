"""Minimal numpy.core shim for local stub.

This exposes the top-level stubbed numpy symbols under numpy.core
so that imports like `import numpy.core` succeed.
"""
from .. import *  # re-export all stubbed numpy symbols
