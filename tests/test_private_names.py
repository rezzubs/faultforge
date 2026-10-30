"""Checks the `_` prefix convention for module-level names in faultforge._impl.

- A name without `_` must be re-exported by a public module or imported by
  another `_impl` module.
- A name with `_` must not be imported by another `_impl` module.

Tests are exempt and may import `_` names.
"""

import ast
from collections.abc import Iterable, Iterator
from pathlib import Path

import faultforge

_PACKAGE_ROOT = Path(faultforge.__file__).parent
_PUBLIC_PATHS = sorted(_PACKAGE_ROOT.glob("*.py"))
_IMPL_PATHS = sorted((_PACKAGE_ROOT / "_impl").rglob("*.py"))


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(_PACKAGE_ROOT.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _defined_names(path: Path) -> Iterator[str]:
    for node in _parse(path).body:
        match node:
            case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef():
                yield node.name
            case ast.Assign():
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        yield target.id
            case ast.AnnAssign(target=ast.Name(id=name)):
                yield name
            case ast.TypeAlias(name=ast.Name(id=name)):
                yield name
            case _:
                pass


def _imported_names(path: Path) -> Iterator[tuple[str, str]]:
    """Yield `(module, name)` for every `from module import name` in `path`."""
    importer = _module_name(path)
    for node in ast.walk(_parse(path)):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 0:
            assert node.module is not None, "absolute imports always name a module"
            module = node.module
        else:
            # A relative import resolves against the importer's package, which
            # for `__init__.py` is the module itself.
            package_parts = importer.split(".")
            if path.name != "__init__.py":
                package_parts.pop()
            if node.level > 1:
                package_parts = package_parts[: -(node.level - 1)]
            if node.module is not None:
                package_parts.append(node.module)
            module = ".".join(package_parts)
        for alias in node.names:
            yield module, alias.name


def _foreign_imports(importers: Iterable[Path]) -> set[tuple[str, str]]:
    """`(module, name)` pairs imported by `importers` from other modules."""
    return {
        (module, name)
        for path in importers
        for module, name in _imported_names(path)
        if module != _module_name(path)
    }


def test_unprefixed_names_are_used_outside_their_module():
    used = _foreign_imports([*_PUBLIC_PATHS, *_IMPL_PATHS])
    violations = [
        f"{_module_name(path)}.{name}"
        for path in _IMPL_PATHS
        for name in _defined_names(path)
        if not name.startswith("_") and (_module_name(path), name) not in used
    ]
    assert not violations, (
        "Only used in their own module, prefix with `_` or re-export:\n"
        + "\n".join(violations)
    )


def test_prefixed_names_are_not_imported_across_impl_modules():
    violations = [
        f"{module}.{name}"
        for module, name in sorted(_foreign_imports(_IMPL_PATHS))
        if module.startswith("faultforge._impl")
        and name.startswith("_")
        and not _is_dunder(name)
    ]
    assert not violations, (
        "Imported by another `_impl` module, remove the `_` prefix:\n"
        + "\n".join(violations)
    )
