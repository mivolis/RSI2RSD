"""Resume a known contiguous prefix using disjoint HTTP ranges; verify official MD5."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import hashlib,json,os,time,urllib.request
from prepare_data import URL,MD5,SIZE

root=Path('data');temp=root/'CIFAR-100-C.tar.download'
prefix=temp.stat().st_size if temp.exists() else 0
assert 0<=prefix<SIZE
fd=os.open(temp,os.O_RDWR|os.O_CREAT,0o600);os.ftruncate(fd,SIZE)
count=4;step=(SIZE-prefix+count-1)//count
ranges=[(a,min(SIZE-1,a+step-1)) for a in range(prefix,SIZE,step)]
(root/'range_download.json').write_text(json.dumps(dict(contiguous_prefix=prefix,ranges=ranges,url=URL)))
def fetch(bounds):
 start,end=bounds;offset=start
 for attempt in range(3):
  try:
   req=urllib.request.Request(URL,headers={'Range':f'bytes={offset}-{end}'})
   with urllib.request.urlopen(req,timeout=90) as r:
    assert r.status==206 and r.headers['Content-Range']==f'bytes {offset}-{end}/{SIZE}'
    while offset<=end:
     data=r.read(min(1024*1024,end-offset+1))
     if not data:raise IOError('truncated HTTP range')
     view=memoryview(data)
     while view:
      n=os.pwrite(fd,view,offset);offset+=n;view=view[n:]
   print(json.dumps(dict(range_start=start,range_end=end,status='complete')),flush=True)
   return
  except Exception:
   if attempt==2:raise
   time.sleep(2)
with ThreadPoolExecutor(max_workers=count) as pool:
 for future in as_completed([pool.submit(fetch,r) for r in ranges]):future.result()
os.fsync(fd);os.close(fd)
h=hashlib.md5()
with temp.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
assert h.hexdigest()==MD5,'official archive checksum mismatch'
os.replace(temp,root/'CIFAR-100-C.tar')
print('Parallel download verified official MD5',flush=True)
