import sys

from encoded_memory.commands import app
from encoded_memory.logging import setup_logging


def main() -> None:
    setup_logging()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
