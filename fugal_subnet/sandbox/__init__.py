"""Client-side interface for the isolated v2 grading service.

The validator may import this package, but it must never receive access to a
container-engine socket.  Only the separately deployed launcher owns that
capability.
"""

from fugal_subnet.sandbox.client import GradingClient, GradingUnavailable

__all__ = ["GradingClient", "GradingUnavailable"]
