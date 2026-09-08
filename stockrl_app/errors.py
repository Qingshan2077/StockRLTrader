"""Transport-independent, safe application errors."""
from typing import Any


class AppError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 422,
                 details: list[dict[str, Any]] | dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details

    def response(self, request_id: str) -> dict[str, Any]:
        return {'error': {'code': self.code, 'message': self.message,
                          'details': self.details}, 'request_id': request_id}
