"""A random sampler used to pick fault locations without repeats."""

from faultforge._rust import Picker

__all__ = ["Picker"]
