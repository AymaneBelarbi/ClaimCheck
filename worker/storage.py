"""Uploaded files on local disk, addressed by claim and content hash."""

import hashlib
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

_SUFFIX = re.compile(r"\.[a-z0-9]{1,10}")


class Storage:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def put(self, claim_id: uuid.UUID, filename: str, data: bytes) -> str:
        """Store `data` and return its key, `{claim_id}/{sha256}{suffix}`.

        The uploaded filename only contributes its extension, and only a plain one, so a
        client-chosen name never becomes part of a path. The same bytes uploaded twice for one
        claim land in the same file.
        """
        suffix = Path(filename).suffix.lower()
        if not _SUFFIX.fullmatch(suffix):
            suffix = ""
        key = f"{claim_id}/{hashlib.sha256(data).hexdigest()}{suffix}"
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written beside the target and renamed, so a reader never sees half a file.
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".upload-")
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(data)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return key

    def get(self, key: str) -> bytes:
        root = self.root.resolve()
        path = (root / key).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"storage key {key!r} points outside the storage root")
        return path.read_bytes()

    def discard(self, claim_id: uuid.UUID) -> None:
        """Remove everything stored for `claim_id`; used when an upload fails before commit."""
        shutil.rmtree(self.root / str(claim_id), ignore_errors=True)
