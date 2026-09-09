"""Loopback-only, read-only monitor, intentionally separate from training lifetime."""
import argparse,functools,json,os
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control','no-store');super().end_headers()
    def log_message(self,*args):pass
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--port',type=int,default=18766);a=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),functools.partial(Handler,directory=str(a.run.resolve())))
    (a.run/'viewer.json').write_text(json.dumps(dict(pid=os.getpid(),port=a.port,root=str(a.run.resolve()))))
    server.serve_forever()
if __name__=='__main__':main()
