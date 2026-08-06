"""Validation of oracle action sequences against a domain state machine."""

from aieng.syn_data.synbench.fsm.validator import (
    FSMValidationError,
    validate_actions_against_fsm,
)


__all__ = ["FSMValidationError", "validate_actions_against_fsm"]
