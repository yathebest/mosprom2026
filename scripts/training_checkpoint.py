"""Crash-safe replacement of our own training checkpoints on one filesystem."""
import os
from pathlib import Path
import shutil


def atomic_copy(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.writing')
    with Path(source).open('rb') as incoming, temporary.open('wb') as outgoing:
        shutil.copyfileobj(incoming, outgoing)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    os.replace(temporary, target)


def atomic_torch_save(state, target, previous=False):
    import torch
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.writing')
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if previous and target.exists():
        atomic_copy(target, target.with_name(target.stem + '-previous.pt'))
    os.replace(temporary, target)
