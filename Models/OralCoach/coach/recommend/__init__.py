"""Recommendation system: patterns -> retrieval -> strategy choice -> LLM wording -> did-it-help check."""
from .coach import Coach
from .knowledge import Knowledge
from .patterns import PATTERNS, PatternDetector
from .writer import TipWriter

__all__ = ["Coach", "Knowledge", "PATTERNS", "PatternDetector", "TipWriter"]
