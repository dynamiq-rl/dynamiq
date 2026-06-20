"""Dynamiq error types.

The whole point of Dynamiq is that classic RL bugs become *compile-time* errors
instead of silent training failures. These exceptions are how we report them.
"""


class DynamiqError(Exception):
    """Base class for all Dynamiq errors."""


class DynamiqTypeError(DynamiqError):
    """Raised when an algorithm graph violates an RL typing rule.

    Examples of what this catches *before* a single gradient step:
      * a bootstrap target that still carries gradients (missing stop-grad),
      * off-policy data flowing into an on-policy objective,
      * mixing on-policy and off-policy data in the same signal.
    """


class DynamiqConfigError(DynamiqError):
    """Raised when a node is declared with a missing or contradictory config
    (e.g. a target network with no sync rule)."""
