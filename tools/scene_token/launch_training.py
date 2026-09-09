"""Start the read-only viewer and training supervisor detached from SSH."""
import argparse,json,os,socket,subprocess,sys
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    root=a.run.resolve();repo=Path(__file__).resolve().parents[2]
    assert (root/'config.json').exists()
    old=json.loads((root/'pipeline.json').read_text()) if (root/'pipeline.json').exists() else {}
    if old and Path(f"/proc/{old['pid']}/cmdline").exists():
        cmd=Path(f"/proc/{old['pid']}/cmdline").read_bytes()
        assert b'run_training.py' not in cmd,'Training supervisor already exists'
    with socket.socket() as sock:viewer_exists=sock.connect_ex(('127.0.0.1',18766))==0
    if viewer_exists:
        assert (root/'viewer.json').exists() and json.loads((root/'viewer.json').read_text())['root']==str(root),'Port belongs to another viewer'
    def launch(script,log,extra=()):
        with (root/log).open('a') as stream:
            return subprocess.Popen([sys.executable,'-u',f'tools/scene_token/{script}','--run',str(root),*extra],cwd=repo,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True).pid
    viewer=None if viewer_exists else launch('serve_training.py','viewer.log')
    supervisor=launch('run_training.py','pipeline.log',['--resume'] if a.resume else [])
    print(json.dumps(dict(supervisor=supervisor,viewer=viewer,run=str(root))))
if __name__=='__main__':main()
