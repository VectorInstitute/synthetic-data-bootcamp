"""Tests for the required-information substring checker."""

from aieng.syn_data.synbench.environment.communicate import CommunicateChecker


def test_communicate_pass():
    """A reply containing every required string scores full marks."""
    score, missing = CommunicateChecker.check(
        ["cancelled"], ["Your order has been cancelled successfully."]
    )
    assert score == 1.0
    assert missing == []


def test_communicate_fail():
    """A reply missing a required string scores zero and reports it."""
    score, missing = CommunicateChecker.check(["cannot be canceled"], ["Sorry, done."])
    assert score == 0.0
    assert missing
