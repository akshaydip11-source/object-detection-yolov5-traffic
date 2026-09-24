"""Single-worker admission controls. Deploy behind TLS; not a distributed limiter."""

from collections import OrderedDict, deque
from threading import BoundedSemaphore
import time

from fastapi import HTTPException
from starlette.responses import JSONResponse
from jose import JWTError, jwt

from backend.app.config import settings

_inference_slot = BoundedSemaphore(1)


def inference_slot():
    # Reject overload rather than accumulating an unbounded inference queue.
    if not _inference_slot.acquire(blocking=False):
        raise HTTPException(
            503, "Inference busy; retry shortly", headers={"Retry-After": "5"}
        )
    try:
        yield
    finally:
        _inference_slot.release()


class RequestGuards:
    def __init__(self, app):
        self.app = app
        self.windows = OrderedDict()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        headers = dict(scope.get("headers", []))
        is_upload = path.startswith("/api/detect/")
        is_auth = path in {"/api/auth/login", "/api/auth/register"}
        if is_upload:
            # Reject missing/invalid credentials before multipart parsing/spooling.
            # Full active-user/role authorization still happens in the route dependency.
            try:
                scheme, credential = (
                    headers.get(b"authorization", b"").decode("ascii").split(" ", 1)
                )
                payload = jwt.decode(
                    credential, settings.secret_key, algorithms=[settings.algorithm]
                )
                if (
                    scheme.lower() != "bearer"
                    or payload.get("purpose") != "access"
                    or not payload.get("sub")
                ):
                    raise ValueError("Invalid access token")
            except (JWTError, ValueError, UnicodeError):
                return await JSONResponse({"detail": "Authentication required"}, 401)(
                    scope, receive, send
                )
        if is_upload or is_auth:
            # Uses the ASGI peer address. Do not blindly trust arbitrary X-Forwarded-For.
            ip = (scope.get("client") or ("unknown", 0))[0]
            key = (ip, "auth" if is_auth else "detect")
            limit = (
                settings.login_requests_per_minute
                if is_auth
                else settings.detection_requests_per_minute
            )
            now = time.monotonic()
            window = self.windows.setdefault(key, deque())
            self.windows.move_to_end(key)
            while window and now - window[0] >= 60:
                window.popleft()
            if len(window) >= limit:
                return await JSONResponse(
                    {"detail": "Too many requests; retry in a minute"},
                    429,
                    headers={"Retry-After": "60"},
                )(scope, receive, send)
            window.append(now)
            while len(self.windows) > 4096:
                self.windows.popitem(last=False)
            # Includes multipart overhead; the file itself has a separate exact limit.
            maximum = (
                (settings.max_upload_mb + 1) * 1024 * 1024 if is_upload else 16 * 1024
            )
            try:
                declared = int(headers.get(b"content-length", b"0"))
                if declared < 0:
                    raise ValueError()
            except ValueError:
                return await JSONResponse({"detail": "Invalid Content-Length"}, 400)(
                    scope, receive, send
                )
            if declared > maximum:
                return await JSONResponse({"detail": "Request body too large"}, 413)(
                    scope, receive, send
                )
            total = 0
            original_receive = receive

            async def limited_receive():
                nonlocal total
                message = await original_receive()
                total += len(message.get("body", b""))
                if total > maximum:
                    raise HTTPException(413, "Request body too large")
                return message

            receive = limited_receive

        async def guarded_send(message):
            if message["type"] == "http.response.start":
                h = list(message.get("headers", []))
                h.extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"referrer-policy", b"no-referrer"),
                        (
                            b"permissions-policy",
                            b"camera=(self), microphone=(), geolocation=()",
                        ),
                    ]
                )
                if path.startswith("/api/"):
                    h.append((b"cache-control", b"no-store"))
                message = {**message, "headers": h}
            await send(message)

        await self.app(scope, receive, guarded_send)
