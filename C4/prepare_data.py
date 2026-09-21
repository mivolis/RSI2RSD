import hashlib
import json
import os
from pathlib import Path
import tarfile
import urllib.request
import argparse

URL='https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1'
MD5='11f0ed0f1191edbf9fa23466ae6021d3'
SIZE=2918473216


def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    archive=root/'CIFAR-100-C.tar'
    if not archive.exists() or archive.stat().st_size!=SIZE:
        temp=root/'CIFAR-100-C.tar.download'
        with urllib.request.urlopen(URL,timeout=120) as src,temp.open('wb') as dst:
            while True:
                chunk=src.read(8*1024*1024)
                if not chunk:break
                dst.write(chunk)
            dst.flush();os.fsync(dst.fileno())
        assert temp.stat().st_size==SIZE
        os.replace(temp,archive)
    h=hashlib.md5()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    assert h.hexdigest()==MD5,'archive integrity failure'
    with tarfile.open(archive) as tar:
        tar.extractall(root,filter='data')
    (root/'data_provenance.json').write_text(json.dumps(dict(url=URL,md5=MD5,bytes=SIZE),indent=2))
    print('Verified and extracted CIFAR-100-C',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();prepare(a.root)
