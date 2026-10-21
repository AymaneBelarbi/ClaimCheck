import hashlib
import uuid
from pathlib import Path

import pytest

from claimcheck.worker.storage import Storage


def test_put_keys_by_claim_and_content(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    claim_id = uuid.uuid4()
    data = b"%PDF-1.7 fake"

    key = storage.put(claim_id, "Repair Invoice.PDF", data)

    assert key == f"{claim_id}/{hashlib.sha256(data).hexdigest()}.pdf"
    assert storage.get(key) == data
    assert (tmp_path / key).read_bytes() == data


def test_same_bytes_twice_is_one_file(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    claim_id = uuid.uuid4()
    first = storage.put(claim_id, "a.png", b"png")
    second = storage.put(claim_id, "copy.png", b"png")
    assert first == second
    assert len(list((tmp_path / str(claim_id)).iterdir())) == 1


def test_odd_suffixes_are_dropped(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    key = storage.put(uuid.uuid4(), "notes.t$t ", b"x")
    assert Path(key).suffix == ""
    assert storage.put(uuid.uuid4(), "no_suffix", b"x").endswith(hashlib.sha256(b"x").hexdigest())


def test_empty_file_is_stored(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    key = storage.put(uuid.uuid4(), "empty.pdf", b"")
    assert storage.get(key) == b""


def test_keys_cannot_leave_the_root(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "root")
    (tmp_path / "secret").write_bytes(b"no")
    with pytest.raises(ValueError, match="outside"):
        storage.get("../secret")


def test_missing_key(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Storage(tmp_path).get(f"{uuid.uuid4()}/nope.pdf")


def test_discard_removes_the_claims_files_only(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    mine, other = uuid.uuid4(), uuid.uuid4()
    storage.put(mine, "a.pdf", b"a")
    kept = storage.put(other, "b.pdf", b"b")

    storage.discard(mine)

    assert not (tmp_path / str(mine)).exists()
    assert storage.get(kept) == b"b"


def test_discard_of_an_unknown_claim_is_harmless(tmp_path: Path) -> None:
    Storage(tmp_path).discard(uuid.uuid4())
