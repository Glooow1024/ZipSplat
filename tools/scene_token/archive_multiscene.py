"""Archive audited pilot evidence without GPU checkpoints or source datasets."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--history', type=Path, required=True)
    p.add_argument('--logs', type=Path, required=True)
    a = p.parse_args()
    root = a.input
    assert json.loads((root / 'train_v1/validation.json').read_text())['passed']
    assert (root / 'train_v1/comparison.html').is_file()
    for folder in ['ddp_smoke_v1', 'train_v1']:
        source = a.history / folder
        for f in source.rglob('*'):
            if f.is_file() and f.suffix in {'.json', '.jsonl', '.py'} and '__pycache__' not in f.parts:
                dst = root / 'history' / a.history.name / folder / f.relative_to(source)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
    names = [
        'multiscene_train_20260908_v1.log', 'multiscene_train_20260908_v2.log',
        'multiscene_check_20260908_v2.log', 'multiscene_report_20260908_v2.log',
        'multiscene_ddp_smoke_20260908_v1.log', 'isolate_heldout_depth_20260908_v1.log',
        'prepare_multiscene_20260908_v1.log', 'prepare_multiscene_20260908_v2.log',
    ]
    for name in names:
        src = a.logs / name
        assert src.is_file(), src
        dst = root / 'logs' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for name in ['multiscene_data', 'prepare_multiscene', 'isolate_heldout_depth',
                 'train_multiscene', 'check_multiscene', 'report_multiscene', 'archive_multiscene']:
        src = Path('tools/scene_token') / f'{name}.py'
        dst = root / 'tools' / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    dst = root / 'SCENE_TOKEN_MULTISCENE_ZH.md'
    shutil.copy2('docs/SCENE_TOKEN_MULTISCENE_ZH.md', dst)
    files = sorted(f for f in root.rglob('*') if f.is_file()
                   and f.suffix not in {'.pt', '.tmp', '.zip', '.pyc'}
                   and f.name not in {'artifact_manifest.json', 'archive.json'}
                   and '__pycache__' not in f.parts)
    manifest = {str(f.relative_to(root)): {'bytes': f.stat().st_size, 'sha256': digest(f)} for f in files}
    (root / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    files.append(root / 'artifact_manifest.json')
    archive = root.with_name(root.name + '_lightweight.zip')
    assert not archive.exists(), archive
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(f, f.relative_to(root))
    result = dict(path=str(archive), bytes=archive.stat().st_size, sha256=digest(archive),
                  manifest_files=len(manifest), checkpoint_included=False)
    (root / 'archive.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
