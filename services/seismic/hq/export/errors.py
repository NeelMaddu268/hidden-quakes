"""Exceptions of the export stage."""


class ExportError(RuntimeError):
    """A run table, a lane dependency or a bundle constraint blocks the export. The message
    says what is missing and, for a lane dependency, who ships it."""
