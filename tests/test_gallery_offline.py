"""Offline mode, stable person keys, moved photos and the operator's error messages."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from vie.gallery import insights as gi
from vie.gallery.offline import is_local


def _unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


# ---- offline -----------------------------------------------------------------


@pytest.mark.parametrize(("host", "local"), [
    ("127.0.0.1", True), ("::1", True), ("localhost", True), ("[::1]", True),
    (None, True), ("127.5.5.5", True),
    ("huggingface.co", False), ("8.8.8.8", False), ("dns.google", False),
    ("192.168.1.10", False),
])
def test_only_this_machine_counts_as_local(host, local) -> None:
    assert is_local(host) is local


def _run(code: str) -> subprocess.CompletedProcess:
    """In a child process, so the socket guard never touches the test runner."""
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          capture_output=True, text=True, timeout=60)


def test_outbound_connections_fail_at_once_and_local_ones_work() -> None:
    result = _run("""
        import socket, threading, time
        from vie.gallery.offline import blocked, go_offline
        go_offline()

        started = time.perf_counter()
        try:
            socket.create_connection(("huggingface.co", 443), timeout=30)
            print("REMOTE-OPEN")
        except OSError as exc:
            print("remote refused", "offline mode" in str(exc))
        print("fast", time.perf_counter() - started < 1.0)

        server = socket.socket(); server.bind(("127.0.0.1", 0)); server.listen(1)
        threading.Thread(target=server.accept, daemon=True).start()
        socket.create_connection(server.getsockname(), timeout=5).close()
        print("local ok")
        print("logged", blocked())
    """)
    assert result.returncode == 0, result.stderr
    assert "remote refused True" in result.stdout
    assert "fast True" in result.stdout
    assert "local ok" in result.stdout
    assert "REMOTE-OPEN" not in result.stdout


def test_offline_sets_every_switch_and_bypasses_the_system_proxy() -> None:
    result = _run("""
        import os
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
        from vie.gallery.offline import OFFLINE_ENV, go_offline
        go_offline()
        print(all(os.environ.get(k) for k in OFFLINE_ENV))
        print(os.environ.get("HTTPS_PROXY"), os.environ.get("NO_PROXY"))
        import urllib.request
        print(urllib.request.proxy_bypass("127.0.0.1"))
    """)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.split()
    assert lines[0] == "True"
    assert lines[1:3] == ["None", "*"]


# ---- people keep their identity across a re-clustering ------------------------


def _person(number: int, *faces: tuple[str, tuple]) -> gi.Person:
    members = [gi.Face(name, box, _unit(1, 0), person=number) for name, box in faces]
    return gi.Person(number, members, _unit(1, 0))


def test_a_person_key_survives_renumbering() -> None:
    before = _person(3, ("a.jpg", (10, 10, 50, 50)), ("b.jpg", (0, 0, 90, 90)))
    key = gi.person_key(before)
    # After an upload the same faces are person 1, and someone else is person 3.
    after = [_person(1, ("a.jpg", (10, 10, 50, 50)), ("b.jpg", (0, 0, 90, 90))),
             _person(3, ("c.jpg", (5, 5, 40, 40)))]
    info = gi.Insights({}, after, "m")
    assert info.find_person(key) is after[0]
    assert info.find_person("missing.jpg|1,2,3,4") is None
    assert info.find_person("not a key") is None


# ---- photos that moved ---------------------------------------------------------


def test_a_moved_photo_is_found_under_todays_gallery(tmp_path, config) -> None:
    from dataclasses import replace

    (tmp_path / "cat.jpg").write_bytes(b"x")
    moved = replace(config, gallery_path=tmp_path)
    assert gi.photo_path(moved, "cat.jpg", r"Z:\gone\cat.jpg") == str(tmp_path / "cat.jpg")
    assert gi.photo_path(moved, "cat.jpg", str(tmp_path / "cat.jpg")) == str(tmp_path / "cat.jpg")


# ---- what the operator reads ---------------------------------------------------


@pytest.mark.parametrize(("error", "expected"), [
    (ValueError("no face detected in the query image"), "چهره"),
    (OSError("cannot identify image file 'x.heic'"), "قابل خواندن نیست"),
    (FileNotFoundError("query image not found: x.jpg"), "پیدا نشد"),
    (RuntimeError("database is locked"), "به‌روزرسانی"),
    (RuntimeError("CUDA out of memory"), "دوباره امتحان"),
    (ValueError("این عکس‌ها از قبل در گالری بودند"), "از قبل در گالری"),
])
def test_errors_reach_the_operator_in_persian(error, expected) -> None:
    from vie.gallery.app import _fa_error

    message = _fa_error(error)
    assert expected in message
    assert not any("a" <= ch.lower() <= "z" for ch in message.replace("jpg", "")
                   .replace("png", "")), message


def test_a_corrupt_query_image_is_caught_before_any_model(tmp_path) -> None:
    from PIL import Image

    from vie.gallery.app import _readable

    good = tmp_path / "good.png"
    Image.new("RGB", (8, 8)).save(good)
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    assert _readable(good)
    assert not _readable(broken)
    assert not _readable(tmp_path / "missing.jpg")
