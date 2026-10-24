"""Error type shared by every file-validation / preprocessing helper.

Rule for the whole ``utils`` package: no raw Pillow or PyMuPDF exception may
escape a public function. Every problem with an uploaded file is reported as
a ``FileValidationError`` so that ``main.py`` can turn it into a clean HTTP
error with ``HTTPException(status_code=e.status_code, detail=e.detail)``.
"""


class FileValidationError(Exception):
    """A problem with an uploaded file.

    Attributes:
        status_code: HTTP status to return to the client.
            400 = corrupt / unreadable / password protected / empty,
            413 = too large (bytes or pixel dimensions),
            415 = unsupported file type.
        detail: Human readable message for the client.
    """

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail

    def __str__(self) -> str:
        return f"[{self.status_code}] {self.detail}"
