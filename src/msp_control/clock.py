from datetime import datetime


class Clock:
    """Provide timestamps for MSP operations."""

    def now(self) -> datetime:
        """Return the current local time."""
        return datetime.now().astimezone()
