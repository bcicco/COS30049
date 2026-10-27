"""Every error goes out as {"detail": "<message>"}, never a stack trace."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)


def validation_message(exc: RequestValidationError) -> str:
    """pydantic's error list -> one readable line, e.g. 'text: String should have ...'"""
    parts = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"][1:])  # drop the leading "body"
        parts.append(f"{field}: {err['msg']}" if field else err["msg"])
    return "; ".join(parts)


async def on_validation(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return JSONResponse({"detail": validation_message(exc)}, status_code=422)


async def on_unexpected(request: Request, _exc: Exception) -> JSONResponse:
    # full traceback goes to the server log, the client only gets a generic message
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse({"detail": "Something went wrong on the server"}, status_code=500)


def install(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, on_validation)
    app.add_exception_handler(Exception, on_unexpected)
