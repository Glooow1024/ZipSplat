"""Detached pipeline: await prepared dataset, warmup, joint train; never auto-retry failures."""
import argparse,errno,fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.scene_token.training_monitor import atomic,publish

def read_preparation_json(path):
    """Tolerate briefly unavailable shared-filesystem metadata, not permanent errors."""
    for attempt in range(3):
        try:return json.loads(path.read_text())
        except FileNotFoundError:
            if attempt==2:return None
        except json.JSONDecodeError:
            if attempt==2:raise
        except OSError as error:
            if error.errno not in {errno.ESTALE,errno.EIO,errno.ETIMEDOUT} or attempt==2:raise
        time.sleep(.05*(attempt+1))

def preparation_state(dataset,previous,wait_started,now=None):
    # Read directly: exists() followed by read_text() races with remote replacement.
    summary=read_preparation_json(dataset/'summary.json')
    if summary is not None:return summary,previous
    current=read_preparation_json(dataset/'progress.json')
    progress=current if current is not None else previous
    checked=time.time() if now is None else now
    last_update=progress.get('updated',wait_started)
    if checked-last_update>300:raise RuntimeError('Data preparation heartbeat stale or unavailable for 5 minutes')
    return None,progress

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();root=a.run
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    conf=json.loads((root/'config.json').read_text());dataset=Path(conf['dataset']);atomic(root/'pipeline.json',dict(pid=os.getpid(),started=time.time(),resume=a.resume))
    progress={};wait_started=time.time()
    while True:
        summary,progress=preparation_state(dataset,progress,wait_started)
        if summary is not None:break
        publish(root,dict(state='preparing_data',phase='data',step=conf.get('continuation',{}).get('parent_step',0),total_steps=conf['total_steps'],updated=time.time(),message=f"数据准备 {progress.get('done',0)}/{progress.get('total',288)} 场景，失败 {progress.get('failed',0)}；完成后自动启动训练"))
        if (root/'STOP').exists():publish(root,dict(state='paused',phase='data',updated=time.time(),message='已取消自动启动；数据转换任务独立继续'));return
        time.sleep(10)
    assert summary['passed'],summary
    if conf.get('continuation') and not (root/'runtime_check.json').exists():
        publish(root,dict(state='checking',phase='preflight',step=50000,total_steps=conf['total_steps'],updated=time.time(),message='数据审计通过，检查优化器恢复与真实采样'))
        with (root/'preflight.log').open('a') as f:
            subprocess.run([sys.executable,'tools/scene_token/check_training_runtime.py','--config',str(root/'config.json'),'--output',str(root/'runtime_check.json')],stdout=f,stderr=subprocess.STDOUT,check=True)
    resume=None;phase='warmup'
    if a.resume:
        assert not (root/'STOP').exists(),'Remove STOP only when intentionally resuming'
        idx=json.loads((root/'checkpoints.json').read_text());resume=Path(idx['latest']);record=next(r for r in idx['records'] if r['path']==str(resume));step=record['step'];phase=record['phase']
        archive=root/'recovery_history'/str(int(time.time()));archive.mkdir(parents=True)
        for path in [root/'metrics.jsonl',*(root.glob('rank*.jsonl'))]:
            if path.exists():
                shutil.copy2(path,archive/path.name)
                kept=[line for line in path.read_text().splitlines() if json.loads(line)['step']<=step]
                path.write_text('\n'.join(kept)+'\n')
        if (root/'evaluations.json').exists():
            shutil.copy2(root/'evaluations.json',archive/'evaluations.json');atomic(root/'evaluations.json',[e for e in json.loads((root/'evaluations.json').read_text()) if e['step']<=step])
        for path in (root/'renders').glob('step_*'):
            if int(path.name.split('_')[1])>step:path.rename(archive/path.name)
        for path in root.glob('error_rank*.json'):path.rename(archive/path.name)
        if phase=='warmup' and step==conf['stages']['warmup']['end_step']:phase='joint'
    else:
        assert not (root/'metrics.jsonl').exists(),'Existing training requires --resume'
        if conf.get('continuation'):phase='joint'
    for current in (['warmup','joint'] if phase=='warmup' else ['joint']):
        if (root/'STOP').exists():return
        provenance=json.loads((root/'provenance.json').read_text())
        changed=[name for name,digest in provenance['source_sha256'].items()
                 if not Path(name).exists() or hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest]
        assert not changed,'Source changed after run was frozen: '+str(changed)
        log=root/f'{current}_{int(time.time())}.log'
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','--','tools/scene_token/train_main.py','--run',str(root),'--phase',current]
        if resume:cmd+=['--resume',str(resume)]
        elif conf.get('continuation'):
            from tools.scene_token.prepare_continuation_data import digest
            checkpoint=Path(conf['continuation']['checkpoint']);assert digest(checkpoint)==conf['continuation']['checkpoint_sha256'],'Parent checkpoint changed'
            cmd+=['--initialize-from',str(checkpoint)]
        env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
        with log.open('a') as f:
            child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            atomic(root/'active_process.json',dict(pid=child.pid,phase=current,log=str(log),started=time.time()))
            code=child.wait()
        if code:
            errors=[json.loads(p.read_text())['error'] for p in sorted(root.glob('error_rank*.json'))]
            raise RuntimeError(f'Training process exited {code}; log: {log}\n'+('\n'.join(errors) if errors else log.read_text()[-7000:]))
        status=json.loads((root/'status.json').read_text())
        if status['state']=='paused':return
        assert status['state'] in ['phase_complete','complete'],status
        resume=Path(json.loads((root/'checkpoints.json').read_text())['latest'])
    print('Training completed; viewer remains available.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
        root=Path(sys.argv[sys.argv.index('--run')+1]);old=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
        publish(root,dict(old,state='failed',updated=time.time(),error=error));raise
