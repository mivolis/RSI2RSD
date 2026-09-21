"""Verify the completed local Sapelo mirror and package all retained job evidence."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
from datetime import datetime, timezone


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main(experiment):
    e = Path(experiment).resolve()
    root = e / 'Sapelo/job48416583'
    out = e / 'Delivery/sapelo_raw'
    out.mkdir(exist_ok=True)
    status_file = out / 'packaging_status.json'
    archive = out / 'c4_sapelo_job48416583_raw.tar.gz'
    assert not archive.exists(), 'Preserve existing packaging attempts.'
    assert json.loads((root / 'status.json').read_text())['status'] == 'complete'
    save(status_file, {'status': 'verifying_mirror'})
    runs = []
    for b in [5, 15]:
        for horizon in [20, 94, 1410]:
            d = root / f'after{b}_h{horizon}'
            marker = json.loads((d / 'complete.json').read_text())
            verification = json.loads((d / 'verification.json').read_text())
            assert marker['status'] == 'complete'
            assert verification['status'] == 'verified' and verification['horizon'] == horizon
            for name, expected in marker['files'].items():
                assert digest(d / name) == expected, str(d / name)
            for branch in ['reject_closed', 'accept_closed', 'accept_replay']:
                count = len(list((d / branch).glob('step_*.npz')))
                assert count == horizon, (d, branch, count)
            runs.append({'run': d.name, 'hashed_files': len(marker['files']),
                         'raw_future_records': 3 * horizon})
    mirror = {'status': 'verified', 'job': '48416583', 'runs': runs,
              'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'source': '/scratch/yz54720/RSI2RSD/C4/migration_results/job48416583/',
              'scope': 'All six host completion manifests and complete raw step counts; host verification records retained.'}
    save(e / 'Sapelo/mirror_verification.json', mirror)
    print(json.dumps(mirror), flush=True)
    sources = {str(p.relative_to(root.parent)): p for p in root.rglob('*')
               if p.is_file() and '.rsync-partial' not in p.parts and p.suffix != '.tmp'}
    sources['provenance/mirror_verification.json'] = e / 'Sapelo/mirror_verification.json'
    for name, path in {
        'sapelo_short_provenance.tar.gz': e / 'Migration/sapelo_short_provenance.tar.gz',
        'sacct-48416583.txt': e / 'Delivery/sapelo_results/sacct-48416583.txt',
    }.items():
        assert path.is_file(), str(path)
        sources['provenance/' + name] = path
    hashes = {name: digest(path) for name, path in sources.items()}
    save(out / 'SAPELO_RAW_FILES_SHA256.json', hashes)
    save(status_file, {'status': 'packaging', 'files': len(hashes)})
    with archive.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w|') as tf:
            for name, path in sorted(sources.items()):
                tf.add(path, arcname=name, recursive=False)
    save(status_file, {'status': 'roundtrip_verifying', 'files': len(hashes)})
    seen = set()
    with tarfile.open(archive, 'r|gz') as tf:
        for member in tf:
            assert member.isfile() and member.name in hashes and member.name not in seen
            h = hashlib.sha256()
            with tf.extractfile(member) as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    h.update(chunk)
            assert h.hexdigest() == hashes[member.name], member.name
            seen.add(member.name)
    assert seen == set(hashes)
    save(status_file, {'status': 'splitting', 'archive_bytes': archive.stat().st_size})
    whole = hashlib.sha256(); parts = []; part_size = 900 * 1024 * 1024
    with archive.open('rb') as f:
        index = 0
        while f.tell() < archive.stat().st_size:
            part = out / (archive.name + f'.part{index:02d}')
            h = hashlib.sha256(); count = 0
            with part.open('wb') as dst:
                while count < part_size:
                    chunk = f.read(min(8 * 1024 * 1024, part_size - count))
                    if not chunk:
                        break
                    dst.write(chunk); whole.update(chunk); h.update(chunk); count += len(chunk)
            parts.append({'name': part.name, 'bytes': count, 'sha256': h.hexdigest()})
            index += 1
    manifest = {'format': 'Concatenate parts in listed order to restore tar.gz', 'parts': parts,
                'archive_bytes': archive.stat().st_size, 'archive_sha256': whole.hexdigest(),
                'original_files': len(hashes), 'roundtrip_verified': True,
                'job': '48416583', 'origin': mirror['source'],
                'scope': 'All six Sapelo forks including raw predictions, supervision, masks, schedules, full checkpoints and provenance; no credentials/private manuscript.'}
    save(out / 'SAPELO_RAW_ARCHIVE_INDEX.json', manifest)
    (out / 'SAPELO_SHA256SUMS').write_text(''.join(p['sha256'] + '  ' + p['name'] + '\n' for p in parts))
    save(status_file, {'status': 'complete', 'archive_bytes': archive.stat().st_size,
                       'parts': len(parts), 'files': len(hashes), 'archive_sha256': whole.hexdigest()})
    print(json.dumps({'status': 'complete', 'parts': len(parts), 'files': len(hashes)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('experiment')
    args = parser.parse_args()
    try:
        main(args.experiment)
    except Exception as exc:
        out = Path(args.experiment).resolve() / 'Delivery/sapelo_raw'
        out.mkdir(exist_ok=True)
        save(out / 'packaging_status.json', {'status': 'failed', 'error': repr(exc)})
        raise
