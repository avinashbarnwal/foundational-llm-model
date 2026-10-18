"""Foundation model for personalized recommendation.

An autoregressive transformer over tokenized member interaction histories, after
the Netflix Tech Blog post "Foundation Model for Personalized Recommendation".
"""

from netflix_fm.config import Config, load_config

__all__ = ["Config", "load_config"]
