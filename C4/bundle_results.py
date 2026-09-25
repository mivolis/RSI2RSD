"""Bundle verified Vast evidence, validate an archive round-trip, then split for GitHub."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def main(experiment):
    e=Path(experiment).resolve();remote=e/'remote';out=e/'Delivery/raw'
    status=json.loads((e/'supervisor_status.json').read_text())
    assert status['status']=='retrieved_and_destroyed',status
    original=json.loads((e/'remote_file_manifest.json').read_text())
    sources={name:remote/name for name in original}
    for p in remote.iterdir():
        if p.is_file() and p.suffix in ['.py','.sh','.json','.txt','.md']:
            sources[p.name]=p
    for name in ['remote_file_manifest.json','teardown_evidence.json']:
        sources['provenance/'+name]=e/name
    out.mkdir(parents=True,exist_ok=True)
    archive=out/'c4_vast_raw_20260921.tar.gz'
    assert not archive.exists(), 'preserve any previous bundle attempt'
    hashes={}
    for name,path in sources.items():
        h=digest(path)
        if name in original:assert h==original[name],name
        hashes[name]=h
    (out/'RAW_FILES_SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with archive.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0) as gz:
        with tarfile.open(fileobj=gz,mode='w|') as tf:
            for name,path in sorted(sources.items()):tf.add(path,arcname=name,recursive=False)
    seen=set()
    with tarfile.open(archive,'r|gz') as tf:
        for member in tf:
            assert member.isfile() and member.name in hashes,member.name
            h=hashlib.sha256()
            with tf.extractfile(member) as f:
                for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
            assert h.hexdigest()==hashes[member.name],member.name
            seen.add(member.name)
    assert seen==set(hashes)
    parts=[];size=900*1024*1024;whole=hashlib.sha256()
    with archive.open('rb') as f:
        index=0
        while True:
            data=f.read(size)
            if not data:break
            part=out/(archive.name+f'.part{index:02d}');part.write_bytes(data);whole.update(data)
            parts.append(dict(name=part.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()));index+=1
    manifest=dict(format='concatenate parts in listed order to restore tar.gz',parts=parts,
                  archive_sha256=whole.hexdigest(),archive_bytes=archive.stat().st_size,
                  original_files=len(hashes),roundtrip_verified=True,
                  origin='Vast 51934516:/workspace/c4',scope='pilot, H20/H94 forks, preflight, logs, source weights and runtime code; no private source PDFs or credentials')
    (out/'RAW_ARCHIVE_INDEX.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'SHA256SUMS').write_text(''.join(x['sha256']+'  '+x['name']+'\n' for x in parts))
    print(json.dumps({k:v for k,v in manifest.items() if k!='parts'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('experiment');a=p.parse_args();main(a.experiment)
