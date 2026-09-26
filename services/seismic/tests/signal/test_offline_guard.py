"""The suite-wide offline guard (``_no_network`` in conftest.py) itself.

Addresses are documentation ranges (RFC 5737, RFC 3849) and a ``.invalid`` name: even a guard
that let them through would reach nothing.
"""

import socket

import pytest

pytestmark = pytest.mark.smoke

REFUSED = pytest.fail.Exception  # pytest's Failed: a BaseException, not an Exception


def test_a_broad_except_in_the_code_under_test_cannot_swallow_the_refusal() -> None:
    retried: list[Exception] = []

    def retrying_lookup() -> None:
        for _ in range(3):
            try:
                socket.getaddrinfo("example.invalid", 80)
            except Exception as exc:  # noqa: BLE001 - as broad as the downloader's retry loop
                retried.append(exc)

    with pytest.raises(REFUSED, match="network access in an offline test: getaddrinfo"):
        retrying_lookup()
    assert retried == []  # refused on the first attempt, never retried


@pytest.mark.parametrize("host", ["192.0.2.1", "2001:db8::1", "example.invalid"])
def test_remote_create_connection_and_lookup_are_refused(host: str) -> None:
    with pytest.raises(REFUSED, match="create_connection"):
        socket.create_connection((host, 9), timeout=0.1)
    with pytest.raises(REFUSED, match="getaddrinfo"):
        socket.getaddrinfo(host, 9)


def test_remote_socket_connect_is_refused() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.1)
        with pytest.raises(REFUSED, match="connect "):
            sock.connect(("192.0.2.1", 9))
        with pytest.raises(REFUSED, match="connect_ex"):
            sock.connect_ex(("192.0.2.1", 9))


def test_loopback_socketpair_still_works() -> None:
    # asyncio's ProactorEventLoop on Windows needs this for its self-pipe (seisbench's classify)
    a, b = socket.socketpair()
    with a, b:
        a.sendall(b"x")
        assert b.recv(1) == b"x"
