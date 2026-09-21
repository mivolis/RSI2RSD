"""Local execution supervisor: retrieve continuously, enforce time cap, verify before teardown.

Only operates on the instance recorded in this pilot's authorized lease.json.
Never copies API credentials to the GPU host.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from c4 import atomic_json


def supervise(contract):
    contract=Path(contract).resolve();lease=json.loads(contract.read_text())
    assert lease['project']=='RSD_C4_CTTA'
    assert lease['hourly_total']<=1.2 and lease['budget']==10
    root=contract.parent;dest=root/'remote';dest.mkdir(exist_ok=True)
    instance=str(lease['instance_id'])
    ssh=['ssh','-i',str(Path.home()/'.ssh/id_rsa'),'-p',str(lease['ssh_port']),
         '-o','BatchMode=yes','-o','ConnectTimeout=20','-o','StrictHostKeyChecking=accept-new',
         '-o','UserKnownHostsFile='+str(root/'known_hosts'),
         'root@'+lease['ssh_host']]
    cli=str(Path(sys.executable).parent/'vastai')
    shutdown_at=lease['hard_stop_unix']-600
    def remote(cmd):
        return subprocess.run(ssh+[cmd],capture_output=True,text=True,timeout=90)
    def sync():
        cmd=['rsync','-az','--partial-dir=.rsync-partial','--exclude=*.tmp','--exclude=data/','--exclude=.venv/',
             '--exclude=vendor/','--exclude=mplcache/','-e',shlex.join(ssh[:-1]),
             ssh[-1]+':/workspace/c4/',str(dest)+'/']
        try:
            return subprocess.run(cmd,capture_output=True,text=True,timeout=180)
        except subprocess.TimeoutExpired:
            # Slow transport is not an experiment failure. Completed files and
            # partial-file blocks persist; the next pass resumes recovery.
            return subprocess.CompletedProcess(cmd,124,'','sync pass time-limited; partial data retained')
    stopped=False;failures=0
    while True:
        now=time.time()
        if now>=shutdown_at and not stopped:
            r=remote('touch /workspace/c4/STOP')
            stopped=r.returncode==0
        pulled=sync()
        if pulled.returncode:
            failures+=1
            atomic_json(root/'supervisor_status.json',dict(status='retrieval_retry',
                attempts=failures,updated_unix=time.time(),detail=pulled.stderr[-1000:]))
            if time.time()>=shutdown_at+600:
                # Pausing compute preserves its disk for recovery; never destroy unverified artifacts.
                subprocess.run([cli,'stop','instance',instance],capture_output=True,text=True,timeout=60)
                atomic_json(root/'supervisor_status.json',dict(status='attention_compute_stopped_retrieval_failed',
                    updated_unix=time.time(),instance_id=instance,storage_still_billed=True))
                return 2
            time.sleep(45);continue
        failures=0
        subprocess.run([sys.executable,str(dest/'report.py'),str(dest/'results')],
                       capture_output=True,text=True,timeout=120) if (dest/'results').exists() else None
        statuses={}
        for p in (dest/'results').glob('gamma*/status.json'):
            statuses[p.parent.name]=json.loads(p.read_text())
        exits=dest/'worker_exit.json'
        atomic_json(root/'supervisor_status.json',dict(status='retrieving',updated_unix=time.time(),
            instance_id=instance,arms=statuses,stop_requested=stopped,
            elapsed_hours=(time.time()-lease['created_unix'])/3600,
            estimated_time_charge=(time.time()-lease['created_unix'])/3600*lease['hourly_total']))
        if exits.exists():
            # Workers have stopped writing. A final transfer includes stdout tail and completion markers.
            final=sync()
            if final.returncode:continue
            verify=subprocess.run([sys.executable,str(dest/'report.py'),str(dest/'results'),'--verify'],
                                  capture_output=True,text=True,timeout=600)
            (root/'retrieval_verification.log').write_text(verify.stdout+'\n'+verify.stderr)
            if verify.returncode:
                subprocess.run([cli,'stop','instance',instance],capture_output=True,text=True,timeout=60)
                atomic_json(root/'supervisor_status.json',dict(status='attention_verification_failed_compute_stopped',
                    instance_id=instance,storage_still_billed=True,updated_unix=time.time()))
                return 3
            # Hash every retained remote result/log/config, including partial attempts, before destruction.
            command="cd /workspace/c4 && python - <<'PY'\nimport pathlib,hashlib,json\nroot=pathlib.Path('.')\nfiles={}\nfor dirname in ['results','preflight','logs','checkpoints']:\n for p in (root/dirname).rglob('*'):\n  if p.is_file() and not p.name.endswith('.tmp'):\n   h=hashlib.sha256()\n   with p.open('rb') as f:\n    for b in iter(lambda:f.read(1048576),b''):h.update(b)\n   files[str(p)]=h.hexdigest()\nprint(json.dumps(files))\nPY"
            manifest=remote(command)
            if manifest.returncode:continue
            hashes=json.loads(manifest.stdout)
            from c4 import filehash
            missing=[name for name,h in hashes.items() if not (dest/name).is_file() or filehash(dest/name)!=h]
            atomic_json(root/'remote_file_manifest.json',hashes)
            if missing:
                atomic_json(root/'retrieval_mismatch.json',missing);continue
            outcome=json.loads(exits.read_text())
            atomic_json(root/'teardown_evidence.json',dict(instance_id=instance,
                verified_files=len(hashes),worker_exits=outcome,arms=statuses,
                time_unix=time.time(),data_not_retrieved='public CIFAR-100-C; checksum and source retained'))
            result=subprocess.run([cli,'destroy','instance',instance,'--raw'],capture_output=True,text=True,timeout=90)
            (root/'teardown_response.txt').write_text(result.stdout+'\n'+result.stderr)
            check=subprocess.run([cli,'show','instances','--raw'],capture_output=True,text=True,timeout=60)
            try: active=json.loads(check.stdout);gone=all(str(i['id'])!=instance for i in active)
            except Exception:gone=False
            atomic_json(root/'supervisor_status.json',dict(status='retrieved_and_destroyed' if gone else 'attention_verify_teardown',
                instance_id=instance,arms=statuses,worker_exits=outcome,updated_unix=time.time(),
                verified_files=len(hashes),estimated_time_charge=(time.time()-lease['created_unix'])/3600*lease['hourly_total']))
            return 0 if gone else 4
        if time.time()>=shutdown_at+600:
            remote("cd /workspace/c4 && xargs -r kill -TERM < worker_pids.txt")
            subprocess.run([cli,'stop','instance',instance],capture_output=True,text=True,timeout=60)
            atomic_json(root/'supervisor_status.json',dict(status='attention_time_cap_compute_stopped',
                instance_id=instance,storage_still_billed=True,updated_unix=time.time()))
            return 5
        time.sleep(45)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('lease');a=p.parse_args()
    try:
        code=supervise(a.lease)
    except Exception as exc:
        # Fail closed on a controller exception: preserve disk, stop compute, report attention.
        lease=json.loads(Path(a.lease).read_text())
        cli=str(Path(sys.executable).parent/'vastai')
        result=subprocess.run([cli,'stop','instance',str(lease['instance_id'])],
                              capture_output=True,text=True,timeout=60)
        atomic_json(Path(a.lease).parent/'supervisor_status.json',dict(
            status='attention_supervisor_exception',exception=type(exc).__name__,
            detail=str(exc),stop_exit=result.returncode,instance_id=lease['instance_id'],
            storage_still_billed=True,updated_unix=time.time()))
        raise
    sys.exit(code)
