"""Error correction and fault simulation for PyTorch.

Systems
-------
At the core of this library is the :class:`system.BaseSystem` class, which
represents a neural network model and its associated data. The system needs to
expose a core data object and provide a method to get an accuracy metric based
on that object. The system is also responsible for fault injection.

The :class:`stats.Stats` class can be used to record statistics for a system
over many runs of fault injection.

System Implementations
----------------------
- The :mod:`imagenet` package provides a **system** implementation for ImageNet
  models and related types.
- The :mod:`cifar` package provides a **system** implementation for CIFAR models
  and related types.
- The :mod:`encoding` package provides different *encoded* **system** types
  which wrap other **system** implementations.

Utilities
---------
The module :mod:`tensor_ops` provides utility functions for tensors and lists of
tensors including fault injection and bitwise comparison.

"""
