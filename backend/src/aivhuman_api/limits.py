"""Request body size cap, applied before the body is read into memory."""

from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class BodySizeLimit:
    """413 for bodies over max_bytes, whether declared by Content-Length or streamed."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # declared size: reject without reading anything
        length = Headers(scope=scope).get("content-length")
        if length is not None and length.isdigit() and int(length) > self.max_bytes:
            response = JSONResponse({"detail": "Request body too large"}, status_code=413)
            await response(scope, receive, send)
            return

        # chunked bodies carry no length, so count bytes as they arrive
        received = 0

        async def limited() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # fastapi re-raises HTTPException from body parsing, so this becomes a 413
                    raise HTTPException(status_code=413, detail="Request body too large")
            return message

        await self.app(scope, limited, send)
