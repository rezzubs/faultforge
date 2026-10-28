"""Check that the lockstep-versioned packages all match a release tag.

Usage: python scripts/check_versions.py v1.2.3
"""

import sys
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def read_toml(relative_path: str) -> dict:
    with open(REPOSITORY_ROOT / relative_path, "rb") as file:
        return tomllib.load(file)


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <tag>")
    expected = sys.argv[1].removeprefix("v")

    versions = {
        "pyproject.toml": read_toml("pyproject.toml")["project"]["version"],
        "experiments/encoded_memory/pyproject.toml": read_toml(
            "experiments/encoded_memory/pyproject.toml"
        )["project"]["version"],
        "Cargo.toml": read_toml("Cargo.toml")["workspace"]["package"]["version"],
    }

    mismatches = {
        path: version for path, version in versions.items() if version != expected
    }
    for path, version in mismatches.items():
        print(f"{path}: version {version} doesn't match tag {expected}")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
