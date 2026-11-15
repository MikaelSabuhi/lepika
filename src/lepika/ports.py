"""Port probing: can a server bind a port, or is another application on it?"""

from __future__ import annotations

from lepika.system import System

# A server can hold a port on the wildcard address or on loopback alone, and on
# macOS a loopback bind succeeds straight over an existing 0.0.0.0 listener. Both
# are probed, because either one is a real conflict.
# B104 suppressed: these are bind probes, closed immediately without listening.
# Detecting a conflict on the wildcard address requires naming it.
_PROBE_ADDRESSES = ("0.0.0.0", "127.0.0.1")  # nosec B104


def _bind_once(port: int, *, sys: System) -> bool:
    """Is `port` bindable on every address a server might already be holding?"""
    # all() short-circuits: one taken address is enough to call the port busy.
    return all(sys.can_bind(address, port) for address in _PROBE_ADDRESSES)


def port_free(port: int, attempts: int = 3, *, sys: System) -> bool:
    """Can a server bind this port, or is another application already on it?

    Retried, because `lepika update` asks this moments after stopping the server
    that held the port: an OS that has not finished releasing the listening
    socket would otherwise be reported as a port conflict. A free port answers on
    the first try, so only the failing case pays for the wait.
    """
    for attempt in range(attempts):
        if _bind_once(port, sys=sys):
            return True
        if attempt < attempts - 1:
            sys.sleep(0.5)
    return False
