import argparse
import html
import posixpath
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from videosaur.interactive_export import MANIFEST_FILENAME


class ViewerRegistry:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def is_single_viewer(self) -> bool:
        return (self.root / MANIFEST_FILENAME).is_file()

    def list_viewers(self):
        if self.is_single_viewer():
            return [""]
        viewers = []
        for path in sorted(self.root.glob(f"**/{MANIFEST_FILENAME}")):
            rel = path.parent.relative_to(self.root).as_posix()
            viewers.append(rel)
        return viewers

    def resolve_viewer_dir(self, viewer: str = "") -> Path:
        viewer = unquote(viewer or "").strip("/")
        if self.is_single_viewer() and viewer in ("", "."):
            return self.root

        normalized = posixpath.normpath(viewer)
        if normalized in ("", "."):
            normalized = ""
        if normalized.startswith("../") or normalized == ".." or normalized.startswith("/"):
            raise ValueError("Invalid viewer path.")

        viewer_dir = (self.root / normalized).resolve()
        if self.root != viewer_dir and self.root not in viewer_dir.parents:
            raise ValueError("Viewer path escapes server root.")
        return viewer_dir


def create_server(root: Path, host: str = "127.0.0.1", port: int = 8000):
    root = Path(root).resolve()
    registry = ViewerRegistry(root)
    handler = partial(InteractiveViewerHandler, directory=str(root), registry=registry)
    return ThreadingHTTPServer((host, port), handler)


def serve(root: Path, host: str = "127.0.0.1", port: int = 8000):
    server = create_server(root, host=host, port=port)
    address, actual_port = server.server_address
    print(f"Serving VideoSAUR viewer at http://{address}:{actual_port}/")
    try:
        server.serve_forever()
    finally:
        server.server_close()


class InteractiveViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, registry: ViewerRegistry, **kwargs):
        self.viewer_registry = registry
        super().__init__(*args, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self.send_error(
                HTTPStatus.NOT_FOUND,
                "Interactive viewer APIs were replaced by static atlas assets.",
            )
            return
        if parsed.path in ("", "/") and not self.viewer_registry.is_single_viewer():
            self._handle_listing()
            return
        super().do_GET()

    def _handle_listing(self):
        viewers = self.viewer_registry.list_viewers()
        items = "\n".join(
            "<li><a href=\"/{href}/\">{label}</a></li>".format(
                href=quote(viewer),
                label=html.escape(viewer or "."),
            )
            for viewer in viewers
        )
        body = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>VideoSAUR Viewers</title>"
            "<style>"
            "body{font-family:system-ui,sans-serif;padding:24px;"
            "background:#f5f6f1;color:#17202a}"
            "a{color:#167c80;text-decoration:none}"
            "a:hover{text-decoration:underline}"
            "li{margin:8px 0}"
            "</style></head><body>"
            "<h1>VideoSAUR Viewers</h1><ul>"
            f"{items}</ul></body></html>"
        ).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="Serve local VideoSAUR interactive viewers.")
    parser.add_argument("root", help="Viewer directory or an output directory containing viewers.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    args = parser.parse_args()
    serve(Path(args.root), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
