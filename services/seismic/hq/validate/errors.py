"""Exceptions of the validate stage."""


class ValidateError(RuntimeError):
    """A run table, a lane dependency or a rerun result blocks the validation. The message says
    what is missing or wrong and, for a lane dependency, who ships it."""
