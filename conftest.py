"""Hypothesis-Profile für verschiedene Test-Intensitäten."""
import os
from hypothesis import settings, Verbosity

# Lokal: schnell
settings.register_profile(
    "dev",
    max_examples=50,
    deadline=None,
    verbosity=Verbosity.normal,
)

# CI: gründlicher
settings.register_profile(
    "ci",
    max_examples=300,
    deadline=None,
    verbosity=Verbosity.normal,
)

# Fuzz: sehr gründlich (nächtlicher Lauf)
settings.register_profile(
    "fuzz",
    max_examples=3000,
    deadline=None,
    verbosity=Verbosity.normal,
)

settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
