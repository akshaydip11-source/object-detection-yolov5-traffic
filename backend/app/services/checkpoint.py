"""Provision an explicitly configured, trusted checkpoint; never a pretrained fallback."""

import hashlib
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import build_opener, HTTPSHandler, HTTPRedirectHandler


class HTTPSOnlyRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise ValueError("Checkpoint redirects must stay HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def checksum(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def provision_checkpoint(
    target: Path,
    url: str | None,
    expected_sha256: str | None,
    max_bytes: int = 200 * 1024**2,
) -> None:
    """Atomic bounded download with mandatory checksum; existing good files aren't fetched again.

    Only the deployment administrator sets this URL. A matching hash checks integrity,
    not pickle safety or model quality; the administrator must trust its source.
    URLs may contain credentials, so exceptions never include the URL/underlying cause.
    """
    if not url:
        return
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("MODEL_URL must be an HTTPS artifact URL")
    if not expected_sha256 or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("MODEL_URL requires a valid MODEL_SHA256")
    expected = expected_sha256.lower()
    target = Path(target)
    if target.is_file() and checksum(target) == expected:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".checkpoint-", suffix=".partial", delete=False
        ) as output:
            pending = Path(output.name)
            digest, size = hashlib.sha256(), 0
            opener = build_opener(HTTPSHandler(), HTTPSOnlyRedirect())
            deadline = time.monotonic() + 120
            with opener.open(url, timeout=30) as response:
                if urlsplit(response.geturl()).scheme != "https":
                    raise ValueError("Insecure checkpoint response")
                while chunk := response.read(1024 * 1024):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Checkpoint download time limit")
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError("Checkpoint exceeds configured download limit")
                    output.write(chunk)
                    digest.update(chunk)
            if not size or digest.hexdigest() != expected:
                raise ValueError("Checkpoint checksum mismatch or empty response")
        pending.replace(target)
    except Exception:
        # Do not log a signed URL, its query parameters, or any response body.
        raise RuntimeError(
            "Checkpoint provisioning failed. Verify artifact access, size and MODEL_SHA256; existing weights were not replaced."
        ) from None
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
