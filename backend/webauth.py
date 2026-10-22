"""Access control for the dashboard, because it points a camera at minors.

The server had none. Every route -- list the roster, start the camera, stream
live video, delete or add an enrolment -- was open to anyone who could reach
the port, and the shipped container binds 0.0.0.0. On a school network that is
every device on it.

Three separate holes are closed here, and they need three different
mechanisms, which is why this is not simply "add a password":

**A shared token** gates every route. Anyone without it gets 401.

**An Origin check** gates the WebSocket. The same-origin policy does not apply
to WebSocket handshakes, so without this any page a teacher happens to visit
can open a socket to ``127.0.0.1:8000`` and receive the live video stream.
A loopback-only deployment is still vulnerable, which makes this independent
of the token rather than redundant with it.

**A same-origin requirement** on state-changing routes. ``POST
/api/session/start`` takes no body, so a plain cross-origin ``<form>`` submits
to it with no preflight and no token visible to the attacker -- classic CSRF,
and the thing it starts is a camera.

The token is read from ``CLASSGRAPH_TOKEN`` or generated at startup and
printed once. Generating rather than defaulting to something fixed matters: a
default password in a repository is the same as no password, and this is the
kind of software that gets deployed by copying a folder.
"""
from __future__ import annotations

import hmac
import os
import secrets
from urllib.parse import urlsplit

#: Environment variable holding the shared token.
TOKEN_ENV = "CLASSGRAPH_TOKEN"

#: Header the browser sends it in. A custom header also defeats CSRF on its
#: own, because a cross-origin form cannot set one.
TOKEN_HEADER = "X-ClassGraph-Token"

#: Query parameter, for the WebSocket handshake, which cannot carry custom
#: headers from browser JavaScript.
TOKEN_PARAM = "token"


def resolve_token() -> tuple[str, bool]:
    """The token to enforce, and whether it was generated rather than supplied.

    Returns:
        ``(token, generated)``. A generated token should be printed once by
        the caller so the operator can use it; it changes on every restart,
        which is correct for a tool normally run for one lesson.
    """
    supplied = os.environ.get(TOKEN_ENV, "").strip()
    if supplied:
        return supplied, False
    return secrets.token_urlsafe(24), True


def token_ok(presented: str | None, expected: str) -> bool:
    """Constant-time token comparison.

    ``hmac.compare_digest`` rather than ``==`` so the time taken does not
    reveal how much of the token was right.
    """
    if not presented:
        return False
    return hmac.compare_digest(presented, expected)


def origin_allowed(origin: str | None, host_header: str | None,
                   extra: frozenset[str] = frozenset()) -> bool:
    """Whether a request's Origin may talk to this server.

    A missing Origin is allowed, because non-browser clients (curl, the test
    suite, a monitoring probe) do not send one and they are not the threat
    this addresses -- a cross-site request from a browser always carries it.

    Args:
        origin: The ``Origin`` header, if present.
        host_header: The ``Host`` header, used to accept the page's own origin
            without the server needing to know its own address.
        extra: Additional origins to permit, for a separate front end.
    """
    if not origin:
        return True
    netloc = urlsplit(origin).netloc.lower()
    if not netloc:
        return False
    if host_header and netloc == host_header.lower():
        return True
    return netloc in {urlsplit(e).netloc.lower() for e in extra}


#: Shown when a browser reaches a gated URL directly. Deliberately says what
#: to do and nothing about the token itself: anyone who can read this page
#: has not been authenticated.
_LOCKED_PAGE = """<!doctype html>
<meta charset="utf-8"><title>ClassGraph - access token needed</title>
<style>
 body{font:15px/1.6 system-ui,sans-serif;background:#14171a;color:#dfe3e6;
      margin:0;display:grid;place-items:center;min-height:100vh}
 main{max-width:34rem;padding:2rem}
 h1{font-size:1.25rem;margin:0 0 .75rem}
 p{color:#9aa3ab;margin:.5rem 0}
 code{background:#1e2227;padding:.15rem .4rem;border-radius:3px;color:#dfe3e6}
 a{color:#4aa3a2}
</style>
<main>
 <h1>This page needs an access token</h1>
 <p>ClassGraph prints one in the terminal each time it starts. It changes on
    every restart unless <code>CLASSGRAPH_TOKEN</code> is set.</p>
 <p>Open the dashboard at <a href="/">the main page</a> and paste the token
    when it asks. Reports open from there, with the token attached.</p>
</main>
"""


def install(app, token: str, *, extra_origins: frozenset[str] = frozenset()):
    """Attach token and origin enforcement to a FastAPI app.

    Done as middleware rather than per-route dependencies so a route added
    later is protected by default. Forgetting a dependency on one new endpoint
    is exactly how this class of hole reappears.
    """
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import HTMLResponse, JSONResponse

    #: Served without a token so a browser can load the page and then ask for
    #: one. They expose no data and change no state.
    PUBLIC = {"/", "/favicon.ico"}

    class Guard(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            path = request.url.path
            origin = request.headers.get("origin")
            host = request.headers.get("host")

            if not origin_allowed(origin, host, extra_origins):
                return JSONResponse(
                    {"error": "origin not allowed"}, status_code=403)

            if path in PUBLIC or path.startswith("/static/"):
                return await call_next(request)

            presented = (request.headers.get(TOKEN_HEADER)
                         or request.query_params.get(TOKEN_PARAM))
            if not token_ok(presented, token):
                # A person who typed or bookmarked a gated URL gets words; a
                # program gets JSON. Answering a browser with a raw JSON body
                # tells the reader nothing about what to do, and this is a
                # tool whose users are teachers rather than developers.
                if "text/html" in request.headers.get("accept", ""):
                    return HTMLResponse(_LOCKED_PAGE, status_code=401)
                return JSONResponse(
                    {"error": "missing or invalid token",
                     "hint": f"send {TOKEN_HEADER} or ?{TOKEN_PARAM}="},
                    status_code=401)

            # The cross-origin check above already rejected a foreign Origin
            # for every method, so a second check here could never fire. The
            # protection for unsafe methods is that one plus the token, which
            # a cross-origin form cannot set.
            return await call_next(request)

    app.add_middleware(Guard)
    return app


async def websocket_allowed(socket, token: str,
                            extra_origins: frozenset[str] = frozenset()) -> bool:
    """Check a WebSocket handshake before accepting it.

    Middleware does not cover WebSocket scopes in Starlette, so this is called
    explicitly by the route. Returning False means the caller should close
    without accepting -- accepting first and closing after would already have
    sent the handshake a hijacking page needs.
    """
    origin = socket.headers.get("origin")
    host = socket.headers.get("host")
    if not origin_allowed(origin, host, extra_origins):
        return False
    presented = (socket.headers.get(TOKEN_HEADER)
                 or socket.query_params.get(TOKEN_PARAM))
    return token_ok(presented, token)
