"""Local comparison dashboard: python -m method_3.server --port 8765."""
import argparse
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from method_3.comparison import ComparisonEngine
from method_3.native import PACKAGE_DIR
from method_3.sources import DEFAULT_PARSED

DASHBOARD = PACKAGE_DIR / 'dashboard'


def parse_controls(body):
    if not isinstance(body, dict):
        raise ValueError('The request must be a JSON object')
    n, seed, sector = body.get('n', 200), body.get('seed', 42), body.get('sector', 'all')
    if type(n) is not int or not 20 <= n <= 2000:
        raise ValueError('Sample count must be an integer from 20 to 2,000')
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError('Seed must be an integer from 0 to 4294967295')
    if not isinstance(sector, str):
        raise ValueError('Sector must be a string')
    return n, seed, sector


def static_file(path):
    allowed = {'/': DASHBOARD / 'index.html', '/app.js': DASHBOARD / 'app.js',
               '/style.css': DASHBOARD / 'style.css', '/statistics.js': DASHBOARD / 'statistics.js',
               '/vendor/d3.js': DASHBOARD / 'vendor/d3.v7.9.0.min.js'}
    return allowed.get(path)


def make_handler(engine):
    lock = threading.Lock()
    cache = {}

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, data, content_type, status=200):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def send_json(self, value, status=200):
            self.send_bytes(json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8'),
                            'application/json; charset=utf-8', status)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/api/info':
                self.send_json(dict(sectors=list(engine.by_sector), source=engine.provenance))
                return
            file = static_file(path)
            if not file or not file.is_file():
                self.send_json({'error': 'Not found'}, 404)
                return
            content_type = mimetypes.guess_type(str(file))[0] or 'application/octet-stream'
            if file.suffix in ('.html', '.css', '.js'):
                content_type += '; charset=utf-8'
            self.send_bytes(file.read_bytes(), content_type)

        def do_POST(self):
            if urlsplit(self.path).path != '/api/generate':
                self.send_json({'error': 'Not found'}, 404)
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 4096:
                    raise ValueError('Invalid request length')
                body = json.loads(self.rfile.read(length))
                key = parse_controls(body)
                with lock:
                    if key not in cache:
                        result = engine.generate(*key)
                        if len(cache) >= 3:
                            del cache[next(iter(cache))]
                        cache[key] = result
                    result = cache[key]
                self.send_json(result)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.send_json({'error': str(exc)}, 400)
            except Exception as exc:
                self.send_json({'error': f'Generation failed: {type(exc).__name__}: {exc}'}, 500)

        def log_message(self, fmt, *args):
            print(fmt % args, flush=True)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parsed', type=Path, default=DEFAULT_PARSED)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    engine = ComparisonEngine(args.parsed)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(engine))
    print(f'Financial comparison ready: http://127.0.0.1:{args.port} ({len(engine.rows)} eligible real filings)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
