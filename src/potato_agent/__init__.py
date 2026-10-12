import os
import socket

__version__ = "0.1.0"

if os.environ.get("POTATO_FORCE_IPV4") == "1":
    # Some networks have a broken IPv6 route to download hosts (e.g. the
    # HuggingFace CDN) that hangs in SYN-SENT indefinitely instead of
    # failing fast and falling back to IPv4. Forcing AF_INET here means
    # nothing in this process -- stdlib, requests/urllib3, etc. -- can even
    # attempt the broken address. Off by default since most networks don't
    # need this.
    _orig_getaddrinfo = socket.getaddrinfo

    def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = _ipv4_only_getaddrinfo
