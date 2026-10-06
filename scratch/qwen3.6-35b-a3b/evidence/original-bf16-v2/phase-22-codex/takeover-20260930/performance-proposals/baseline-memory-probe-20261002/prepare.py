#!/usr/bin/env python3
"""Prepare source-only isolated overlay. Does not build or load a model."""
import hashlib, json, pathlib, shutil, subprocess
from evidence_helpers import overlay_hashes
BASE = pathlib.Path(__file__).resolve().parent
ROOT = next(p for p in BASE.parents if (p / '.git').exists())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    baseline = json.loads((BASE / 'baseline-hashes.json').read_text())
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() == baseline['commit'], 'commit drift'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(), 'live edits'
    for name, digest in baseline['files'].items(): assert sha(ROOT/name) == digest, 'baseline drift: '+name
    for name, digest in json.loads((BASE/'frozen-hashes.json').read_text()).items(): assert sha(BASE/name) == digest, 'stage drift: '+name
    overlay = BASE/'overlay'
    assert not overlay.exists(), 'overlay already exists; do not overwrite evidence'
    overlay.mkdir()
    shutil.copytree(ROOT/'Sources', overlay/'Sources')
    for name in ['ThirdParty','Tests']:
        (overlay/name).symlink_to(ROOT/name, target_is_directory=True)
    shutil.copy2(ROOT/'Package.resolved',overlay/'Package.resolved')
    for src in (BASE/'candidate').rglob('*'):
        if src.is_file():
            dest=overlay/src.relative_to(BASE/'candidate');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    build = BASE/'build';build.mkdir()
    for name in ['checkouts','repositories','artifacts']:
        original=ROOT/'.build'/name
        assert original.exists(), 'local dependency cache absent: '+name
        (build/name).symlink_to(original,target_is_directory=True)
    shutil.copy2(ROOT/'.build/workspace-state.json',build/'workspace-state.json')
    (BASE/'overlay-prepared-hashes.json').write_text(json.dumps(overlay_hashes(),indent=2)+'\n')
    command=['swift','build','-c','release','--package-path',str(overlay),'--scratch-path',str(build),'--disable-automatic-resolution','--skip-update','--product','TurboFieldfareBaselineMemoryProbe']
    (BASE/'build-command.json').write_text(json.dumps(command,indent=2)+'\n')
    print(json.dumps({'overlay':str(overlay),'buildCommand':command,'modelRun':False}))
if __name__=='__main__':main()
