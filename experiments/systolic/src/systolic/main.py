lazy from systolic.commands import app
lazy from systolic.logging import setup_logging


def main() -> None:
    setup_logging()
    app()


if __name__ == "__main__":
    main()
