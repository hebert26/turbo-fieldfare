#!/usr/bin/env python3
"""Main-only offline build of prepared isolated overlay, no model launch."""
import json,pathlib,subprocess
from evidence_helpers import BASE,sha,overlay_hashes
def main():
    for name,digest in json.loads((BASE/'frozen-hashes.json').read_text()).items():assert sha(BASE/name)==digest,'stage drift: '+name
    command=json.loads((BASE/'build-command.json').read_text())
    assert not (BASE/'build.stdout').exists(),'build evidence already exists'
    before=overlay_hashes();assert before==json.loads((BASE/'overlay-prepared-hashes.json').read_text()),'prepared overlay drift'
    for name in (BASE/'candidate').rglob('*'):
        if name.is_file():assert sha(name)==sha(BASE/'overlay'/name.relative_to(BASE/'candidate')),'candidate/overlay mismatch'
    (BASE/'overlay-before-build-hashes.json').write_text(json.dumps(before,indent=2)+'\n')
    with (BASE/'build.stdout').open('xb') as out,(BASE/'build.stderr').open('xb') as err:process=subprocess.run(command,stdout=out,stderr=err)
    after=overlay_hashes();(BASE/'overlay-after-build-hashes.json').write_text(json.dumps(after,indent=2)+'\n')
    binary=BASE/'build/release/TurboFieldfareExpertPrefetchOverlap'
    receipt={'command':command,'exit':process.returncode,'modelRun':False,'overlayStable':before==after,'overlayHashesSHA256':sha(BASE/'overlay-after-build-hashes.json'),'binarySHA256':sha(binary) if binary.exists() else None,'stageHashesSHA256':sha(BASE/'frozen-hashes.json')}
    (BASE/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return process.returncode if process.returncode else (0 if receipt['overlayStable'] and receipt['binarySHA256'] else 2)
if __name__=='__main__':raise SystemExit(main())
