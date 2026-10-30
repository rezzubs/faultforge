"""`encoded_memory.args` stays independent so it can become its own library."""

import ast
import sys
from pathlib import Path

import encoded_memory.args


def test_only_imports_the_standard_library_and_itself():
    package_directory = Path(encoded_memory.args.__file__).parent
    foreign_imports: list[str] = []
    for module_path in package_directory.rglob("*.py"):
        for node in ast.walk(ast.parse(module_path.read_text())):
            match node:
                case ast.Import(names=names):
                    imported = [alias.name for alias in names]
                case ast.ImportFrom(module=str(module), level=0):
                    imported = [module]
                case _:
                    continue
            foreign_imports.extend(
                f"{module_path.name}: {name}"
                for name in imported
                if name.split(".")[0] not in sys.stdlib_module_names
                and not name.startswith("encoded_memory.args")
            )
    assert foreign_imports == []
