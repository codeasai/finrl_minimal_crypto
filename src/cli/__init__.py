# src/cli/__init__.py - CLI Package
"""
CLI package สำหรับ finrl_minimal_crypto

Contains command-line interface implementations:
- interactive_cli.py: Interactive Command-Line Interface
"""

from .interactive_cli import (
    InteractiveCLI,
    AgentManager,
    DataManager
)

__version__ = "1.0.0"
__author__ = "finrl_minimal_crypto"

__all__ = [
    'InteractiveCLI',
    'AgentManager',
    'DataManager'
] 