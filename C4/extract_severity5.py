"""Lossless transport extraction from the checksum-verified official archive."""
import argparse,json,tarfile
from pathlib import Path
import numpy as np
from c4 import DOMAINS,filehash,atomic_json
from prepare_data import MD5,SIZE,URL
p=argparse.ArgumentParser();p.add_argument('original');p.add_argument('destination');a=p.parse_args()
src=Path(a.original);dst=Path(a.destination);dst.mkdir(parents=True,exist_ok=True)
provenance=json.loads((src.parent/'data_provenance.json').read_text())
assert provenance['md5']==MD5 and provenance['bytes']==SIZE
labels=np.load(src/'labels.npy');base=labels[:10000]
assert all(np.array_equal(base,labels[k:k+10000]) for k in range(0,len(labels),10000))
np.save(dst/'labels.npy',base)
for domain in DOMAINS:
 arr=np.load(src/(domain+'.npy'),mmap_mode='r');assert arr.shape==(50000,32,32,3)
 np.save(dst/(domain+'.npy'),arr[40000:50000])
 assert np.array_equal(np.load(dst/(domain+'.npy'),mmap_mode='r'),arr[40000:50000])
manifest=dict(source=provenance,extraction='exact rows 40000:50000, severity 5; labels first 10000, all severity copies equal',
              files={f.name:filehash(f) for f in dst.glob('*.npy')})
atomic_json(dst/'input_manifest.json',manifest)
with tarfile.open(dst.parent/'severity5_verified.tar.gz','w:gz',compresslevel=1) as t:
 for f in dst.iterdir():t.add(f,arcname='CIFAR-100-C/'+f.name)
print('Lossless extraction verified; transport bytes',(dst.parent/'severity5_verified.tar.gz').stat().st_size,flush=True)
