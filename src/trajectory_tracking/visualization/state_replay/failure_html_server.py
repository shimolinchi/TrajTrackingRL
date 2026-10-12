"""CPU-only recorded policy state replay, reusing the project's Three.js renderer."""
from __future__ import annotations
import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs, unquote
from trajectory_tracking.visualization.browser import _THREE_RENDERER_MODULE

HERE = Path(__file__).resolve().parent
PREFIX = '/evaluation-review/'


def make_handler(data: Path):
    def cases():
        return {p.stem: p for p in sorted(data.glob('*.json')) if p.stem != 'catalog'}

    class Handler(BaseHTTPRequestHandler):
        def send(self, content, mime='application/json', status=200):
            payload = content.encode() if isinstance(content, str) else content
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            request = urlsplit(self.path)
            path = unquote(request.path)
            if not path.startswith(PREFIX) and path != '/evaluation-review':
                path = PREFIX + path.lstrip('/')
            available = cases()
            if path in ('/', '/evaluation-review'):
                self.send_response(302)
                self.send_header('Location', PREFIX)
                self.end_headers()
                return
            if path == PREFIX + 'player.js':
                return self.send((HERE / 'failure_html_player.js').read_bytes(), 'text/javascript')
            if path.startswith(PREFIX + 'assets/'):
                suffix = path[len(PREFIX + 'assets/'):]
                case_id, _, asset_path = suffix.partition('/')
                if case_id not in available:
                    return self.send('not found', 'text/plain', 404)
                item = json.loads(available[case_id].read_text())
                asset = item['assets'].get('/visualization-assets/' + asset_path)
                if asset is None:
                    return self.send('not found', 'text/plain', 404)
                return self.send(Path(asset['local_path']).read_bytes(), asset['content_type'])
            if path == PREFIX + 'frames':
                case_id = parse_qs(request.query).get('case', [''])[0]
                if case_id not in available:
                    return self.send('not found', 'text/plain', 404)
                item = json.loads(available[case_id].read_text())
                return self.send(json.dumps(item['frames']))
            if path != PREFIX:
                return self.send('not found', 'text/plain', 404)
            if not available:
                return self.send('<meta charset="utf-8"><h2>评估案例正在准备，请稍后刷新</h2>', 'text/html; charset=utf-8')
            case_id = parse_qs(request.query).get('case', [next(iter(available))])[0]
            if case_id not in available:
                return self.send('unknown case', 'text/plain', 404)
            item = json.loads(available[case_id].read_text())
            metadata = {k: v for k, v in item.items() if k not in ('frames', 'scene', 'assets')}
            scene = json.dumps(item['scene']).replace('/visualization-assets/', 'assets/' + case_id + '/')
            labels = {k: f'实验 {k.split("_")[0][3:]} · 案例 {sum(x.startswith(k.split("_")[0]+"_") for x in list(available)[:i+1])}（test{k.split("test")[-1]}）' for i,k in enumerate(available)}
            options = ''.join(f'<option value="{html.escape(k)}" {"selected" if k == case_id else ""}>{html.escape(labels[k])}</option>' for k in available)
            safe = lambda s: s.replace('</', '<\\/')
            page = (HERE / 'failure_html_page.html').read_text()
            page = page.replace('__OPTIONS__', options).replace('__META__', safe(json.dumps(metadata)))
            page = page.replace('__SCENE__', safe(scene)).replace('__RENDERER__', _THREE_RENDERER_MODULE)
            return self.send(page, 'text/html; charset=utf-8')

        def log_message(self, format, *args):
            if args and str(args[1] if len(args) > 1 else '') != '200':
                super().log_message(format, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8392)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(args.data))
    print(f'HTML replay ready: http://127.0.0.1:{args.port}{PREFIX}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
