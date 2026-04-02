"""Error taxonomy for the YouTube draft uploader."""

from __future__ import annotations


class UploaderError(Exception):
    """Base uploader error."""


class ConfigError(UploaderError):
    """Raised for invalid uploader configuration."""


class StateError(UploaderError):
    """Raised for state persistence problems."""


class MetadataError(UploaderError):
    """Raised when a prepared upload folder is invalid."""


class AuthenticationError(UploaderError):
    """Raised when OAuth authentication cannot be completed."""


class DuplicateUploadError(UploaderError):
    """Raised when the same job is detected as already uploaded."""


class RetryableApiError(UploaderError):
    """Raised for transient API failures."""


class PermanentApiError(UploaderError):
    """Raised for non-retryable API failures."""
