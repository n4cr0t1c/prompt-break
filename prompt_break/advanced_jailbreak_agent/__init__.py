"""Advanced Jailbreak Agent package (moved under prompt_break/).

This mirrors the previous top-level package location so tools that import
`advanced_jailbreak_agent` inside the package continue to work via
`prompt_break.advanced_jailbreak_agent`.
"""

from .init import AdvancedJailbreakAgent, AnalysisResult, AttackClass  # re-export if present

__all__ = [
    "AdvancedJailbreakAgent",
    "AnalysisResult",
    "AttackClass",
]
