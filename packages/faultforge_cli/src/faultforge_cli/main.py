import typer

app = typer.Typer()


@app.command()
def main() -> None:
    pass


if __name__ == "__main__":
    app()
