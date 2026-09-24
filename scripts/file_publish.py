"""Publish staged local bundles without replacing an existing destination."""

from pathlib import Path
import shutil


def publish_directory(source, destination):
    """Reserve a new directory atomically; clean up our own copy on failure.

    Copying is not atomically visible to concurrent readers, but competing writers
    cannot replace an existing directory, even an empty one. Staging stays intact.
    """
    destination = Path(destination)
    destination.mkdir(mode=0o700)  # Exclusive reservation; never delete on failure here.
    try:
        shutil.copytree(source, destination, dirs_exist_ok=True)
    except BaseException:
        shutil.rmtree(destination)
        raise
