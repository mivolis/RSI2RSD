"""Independent local price/time guard; only stops the recorded C4 lease."""
import json,subprocess,sys,time
from pathlib import Path
p=Path(sys.argv[1]);lease=json.loads(p.read_text())
assert lease['project']=='RSD_C4_CTTA' and lease['hourly_total']<=1.2 and lease['budget']==10
cli=str(Path(sys.executable).parent/'vastai')
while time.time()<lease['hard_stop_unix']:
    if (p.parent/'teardown_evidence.json').exists():
        status=p.parent/'supervisor_status.json'
        if status.exists() and json.loads(status.read_text()).get('status')=='retrieved_and_destroyed':sys.exit(0)
    time.sleep(30)
r=subprocess.run([cli,'stop','instance',str(lease['instance_id'])],capture_output=True,text=True,timeout=90)
(p.parent/'hard_stop_guard.json').write_text(json.dumps({'instance_id':lease['instance_id'],'stop_returncode':r.returncode,'time':time.time(),'storage_still_billed':True}))
