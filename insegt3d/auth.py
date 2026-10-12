import secrets
from urllib.parse import urlencode

from starlette.requests import HTTPConnection
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.websockets import WebSocketClose

LOGIN_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Interactive 3D Segmentation</title>
<style>
body { font-family: sans-serif; background: #f5f5f5; display: flex; justify-content: center; padding-top: 15vh; margin: 0 }
form { background: white; padding: 24px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2); width: 280px }
h1 { font-size: 18px; margin: 0 0 16px }
input { box-sizing: border-box; width: 100%; padding: 8px; margin-bottom: 12px; font-size: 14px }
button { width: 100%; padding: 8px; font-size: 14px; cursor: pointer }
p { color: #c62828; font-size: 13px; margin: 0 0 12px }
</style>
</head>
<body>
<form>
<h1>InSegt3D</h1>
{error}
<input type="password" name="token" placeholder="Token" autofocus>
<button type="submit">Log in</button>
</form>
</body>
</html>"""


class TokenAuth:
    """ASGI middleware that only lets requests through once they present the token.

    The token is accepted as a `token` query parameter, as an `Authorization: token ...` or
    `Authorization: Bearer ...` header, or as a cookie. A valid query parameter sets the cookie.
    """

    def __init__(self, asgi_app, token: str, cookie_name: str):
        self.asgi_app = asgi_app
        self.token = token.encode()
        self.cookie_name = cookie_name

    def _valid(self, value: str | None) -> bool:
        return value is not None and secrets.compare_digest(value.encode(), self.token)

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.asgi_app(scope, receive, send)
            return

        conn = HTTPConnection(scope)
        query_token = conn.query_params.get("token")
        scheme, _, header_token = conn.headers.get("authorization", "").partition(" ")
        query_ok = self._valid(query_token)
        header_ok = scheme.lower() in ("token", "bearer") and self._valid(header_token)
        cookie_ok = self._valid(conn.cookies.get(self.cookie_name))

        if query_ok and scope["type"] == "http" and scope["method"] == "GET":
            await self._set_cookie_and_redirect(conn, scope, receive, send)
        elif query_ok or header_ok or cookie_ok:
            await self.asgi_app(scope, receive, send)
        elif scope["type"] == "websocket":
            await WebSocketClose(code=1008)(scope, receive, send)
        else:
            error = '<p>Invalid token</p>' if query_token is not None else ''
            response = HTMLResponse(LOGIN_PAGE.replace("{error}", error), status_code=401)
            await response(scope, receive, send)

    async def _set_cookie_and_redirect(self, conn, scope, receive, send):
        """Redirects to the same page without the token in the URL, relative so that any proxy prefix still works."""
        query = urlencode([(k, v) for k, v in conn.query_params.multi_items() if k != "token"])
        location = scope["path"].rsplit("/", 1)[-1] or "./"
        response = RedirectResponse(location + (f"?{query}" if query else ""), status_code=303)
        response.set_cookie(
            self.cookie_name, self.token.decode(), path=scope.get("root_path") or "/", httponly=True, samesite="lax"
        )
        await response(scope, receive, send)
