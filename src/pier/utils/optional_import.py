"""Helpers for environment integration dependencies.

The utilities here produce clear, actionable error messages when a user tries
to use a feature that requires a missing provider package.
"""

from pier.constants import PYPI_PACKAGE_NAME


class MissingExtraError(ImportError):
    """Raised when an optional dependency is not installed.

    Parameters
    ----------
    package:
        The PyPI package name that is missing (e.g. ``"daytona"``).
    extra:
        The future ``datacurve-pier`` extra that provides this package
        (e.g. ``"daytona"``).
    hint:
        Optional provider-specific guidance appended to the message (e.g.
        install steps for an SDK that is not distributed on PyPI).
    """

    def __init__(self, *, package: str, extra: str, hint: str | None = None) -> None:
        self.package = package
        self.extra = extra
        message = (
            f"The '{package}' package is required but not installed. "
            f"Install it with:\n"
            f"  pip install {PYPI_PACKAGE_NAME}\n"
            f"  uv tool install {PYPI_PACKAGE_NAME}"
        )
        if hint:
            message = f"{message}\n{hint}"
        super().__init__(message)
