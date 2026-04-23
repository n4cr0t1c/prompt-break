"""PROMPT-BREAK package

Lightweight package wrapper exposing the CLI entrypoint.
"""
__version__ = "0.1.0"

from .cli import main  # re-export main for easy invocation

__all__ = ["main", "__version__"]
