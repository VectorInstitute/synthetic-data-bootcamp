"""Checks that an agent's replies mention every required piece of information."""


class CommunicateChecker:
    """Substring match for required communicate_info strings."""

    @staticmethod
    def check(
        required: list[str], agent_messages: list[str]
    ) -> tuple[float, list[str]]:
        """Score the agent messages and return ``(score, missing_strings)``."""
        if not required:
            return 1.0, []
        combined = " ".join(agent_messages).lower()
        missing = [s for s in required if s.lower() not in combined]
        if missing:
            return 0.0, missing
        return 1.0, []
