"""Small, dependency-free helpers for reproducible dataset assembly."""

from pathlib import Path


def sync_image_links(directory, sources):
    """Make a dedicated image directory exactly match ``name -> source``.

    Validate every source and existing entry before changing anything. Only symlinks
    are managed; ordinary files/directories are never deleted or overwritten.
    """
    directory = Path(directory)
    if not sources:
        raise ValueError('dataset contains no images; check the dataset root and scene names')
    resolved = {}
    for name, source in sources.items():
        if Path(name).name != name or name in ('', '.', '..'):
            raise ValueError(f'invalid flat image name: {name!r}')
        source = Path(source).resolve(strict=True)
        if not source.is_file():
            raise ValueError(f'image is not a file: {source}')
        resolved[name] = source
    if directory.is_symlink():
        raise ValueError(f'refusing to manage a symlinked image directory: {directory}')
    if directory.exists():
        for entry in directory.iterdir():
            if not entry.is_symlink():
                raise FileExistsError(f'refusing to replace unmanaged dataset entry: {entry}')
    directory.mkdir(parents=True, exist_ok=True)
    for entry in directory.iterdir():
        if entry.name not in resolved or entry.resolve() != resolved[entry.name]:
            entry.unlink()
    for name, source in resolved.items():
        entry = directory / name
        if not entry.is_symlink():
            entry.symlink_to(source)
    return str(directory)
