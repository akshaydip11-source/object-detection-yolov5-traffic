"""Create an ignored .env with a strong secret. Never overwrites an existing file."""

from pathlib import Path
import secrets

ROOT = Path(__file__).resolve().parents[1]


def main():
    content = (
        (ROOT / ".env.example")
        .read_text()
        .replace(
            "replace-with-a-random-secret-before-sharing", secrets.token_urlsafe(48)
        )
    )
    try:
        with (ROOT / ".env").open("x") as output:
            output.write(content)
        (ROOT / ".env").chmod(0o600)
    except FileExistsError:
        raise SystemExit(
            ".env already exists; left unchanged. Check its secret and deployment settings manually."
        )
    print(
        "Created private .env with a random secret. No secret was printed. Supply trusted models/best.pt next."
    )


if __name__ == "__main__":
    main()
