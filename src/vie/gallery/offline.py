"""Guaranteed offline operation.

Every model is on disk after the first run, yet several libraries still reach for the
network on their own: the Hugging Face hub checks for newer revisions, Gradio reports
analytics, and others look for updates. On a slow or absent connection those calls do
not fail, they wait, and one of them held the gallery's start back for ten minutes.

:func:`go_offline` turns all of that off at the source and then refuses any outbound
connection that is not to this machine, so a call nothing switched off fails at once
instead of hanging. Each blocked destination is logged once.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import threading
from typing import Any

from vie.logging_setup import get_logger

log = get_logger("gallery.offline")

#: Set before the libraries are imported, since several read them only once.
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "GRADIO_ANALYTICS_ENABLED": "False",
    "YOLO_OFFLINE": "True",
    # Ultralytics otherwise pip-installs a missing package the first time it meets a
    # file it cannot read; that must never happen during a presentation.
    "YOLO_AUTOINSTALL": "False",
    "DO_NOT_TRACK": "1",
}
PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                   "http_proxy", "https_proxy", "all_proxy")

_installed = False
_blocked: set[str] = set()
_lock = threading.Lock()


def is_local(host: Any) -> bool:
    """True for this machine: loopback addresses, ``localhost`` and Unix sockets."""
    if host is None:
        return True
    text = str(host).strip("[]").lower()
    if text in {"localhost", "", "0.0.0.0", "::"}:
        return True
    try:
        return ipaddress.ip_address(text.split("%")[0]).is_loopback
    except ValueError:
        return text.endswith(".localhost")


def _host_of(address: Any) -> Any:
    if isinstance(address, tuple) and address:
        return address[0]
    if isinstance(address, (str, bytes)):  # AF_UNIX path
        return None
    return address


def _refuse(host: Any) -> None:
    with _lock:
        if host not in _blocked:
            _blocked.add(host)
            log.info("offline: blocked a connection to %s", host)
    raise OSError(f"offline mode: no connection to {host}")


def go_offline() -> None:
    """Switch every known network feature off and refuse outbound connections."""
    global _installed
    for key, value in OFFLINE_ENV.items():
        os.environ.setdefault(key, value)
    # Python reads the Windows system proxy (here a VPN on 127.0.0.1:10808) whenever no
    # proxy variable is set, and would send even this page's own requests through it,
    # which fails whenever that VPN is off. With NO_PROXY=* nothing uses a proxy, and
    # outbound connections then meet the guard below and fail at once.
    for key in PROXY_VARIABLES:
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"
    if _installed:
        return

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def connect(self: socket.socket, address: Any) -> Any:
        host = _host_of(address)
        if self.family in (socket.AF_INET, socket.AF_INET6) and not is_local(host):
            _refuse(host)
        return original_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> Any:
        host = _host_of(address)
        if self.family in (socket.AF_INET, socket.AF_INET6) and not is_local(host):
            _refuse(host)
        return original_connect_ex(self, address)

    original_getaddrinfo = socket.getaddrinfo

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        # Refusing at name lookup is what makes a blocked call fail in microseconds:
        # a DNS query on a dead link can itself wait for many seconds.
        if not is_local(host.decode() if isinstance(host, bytes) else host):
            _refuse(host)
        return original_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
    socket.getaddrinfo = getaddrinfo  # type: ignore[assignment]
    _installed = True
    log.info("offline mode on: only connections to this machine are allowed")


def blocked() -> list[str]:
    """Destinations refused so far, for diagnosis."""
    with _lock:
        return sorted(str(h) for h in _blocked)
