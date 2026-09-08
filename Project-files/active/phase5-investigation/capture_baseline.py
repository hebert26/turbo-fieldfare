"""Read-only provenance collection. Writes only beside this script."""
import datetime
import hashlib
import json
import pathlib
import subprocess

OUT = pathlib.Path(__file__).resolve().parent / "baseline"
OUT.mkdir(exist_ok=True)
SOURCE = pathlib.Path('/Users/dev-machine/dev/turbo-fieldfare-personal')
WORKSPACE = pathlib.Path(__file__).resolve().parent.parent
records = []

def command(name, argv, cwd=SOURCE):
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    p = subprocess.run(argv, cwd=cwd, capture_output=True)
    (OUT / (name + '.stdout')).write_bytes(p.stdout)
    (OUT / (name + '.stderr')).write_bytes(p.stderr)
    records.append(dict(name=name, argv=argv, cwd=str(cwd), started_utc=start,
                        exit_code=p.returncode, stdout_sha256=hashlib.sha256(p.stdout).hexdigest()))
    return p.stdout

command('source-head', ['git', 'rev-parse', 'HEAD'])
command('source-status', ['git', 'status', '--short', '--untracked-files=all'])
command('source-diff-stat', ['git', 'diff', '--stat', 'HEAD'])
diff = command('source-tracked-diff', ['git', 'diff', '--binary', 'HEAD'])
command('workspace-head', ['git', 'rev-parse', 'HEAD'], WORKSPACE)
command('macos', ['sw_vers'])
command('swift', ['swift', '--version'])
command('hardware', ['sysctl', 'hw.model', 'hw.memsize', 'hw.ncpu', 'machdep.cpu.brand_string'])
command('disk', ['df', '-h', str(SOURCE), '/Users/dev-machine/Library/Application Support/TurboFieldfare/gemma4.gturbo'])
command('power', ['pmset', '-g', 'custom'])
command('power-source', ['pmset', '-g', 'batt'])
paths = command('source-git-files', ['git', 'ls-files', '-co', '--exclude-standard', '-z']).decode().split('\0')
relevant = sorted({p for p in paths if p and (p.startswith(('Sources/', 'Scripts/', 'script/')) or p in ('Package.swift', 'Package.resolved', 'AGENTS.md'))})
hashes = []
for relative in relevant:
    p = SOURCE / relative
    if p.is_file():
        b = p.read_bytes()
        hashes.append(dict(path=relative, bytes=len(b), sha256=hashlib.sha256(b).hexdigest(), mtime_ns=p.stat().st_mtime_ns))
    else:
        hashes.append(dict(path=relative, absent=True))
(OUT / 'source-files.json').write_text(json.dumps(hashes, indent=2) + '\n')

app = pathlib.Path('/Applications/TurboFieldfare.app/Contents')
files = [app / 'Info.plist', app / 'MacOS/TurboFieldfareMac', app / 'MacOS/TurboFieldfareDecodeService',
         SOURCE / '.build/arm64-apple-macosx/release/TurboFieldfareMac',
         SOURCE / '.build/arm64-apple-macosx/release/TurboFieldfareDecodeService']
files += sorted((app / 'Resources').rglob('*.metal'))
model = pathlib.Path('/Users/dev-machine/Library/Application Support/TurboFieldfare')
files += [model / 'gemma4.gturbo/manifest.json', model / 'gemma4.gturbo/tokenizer/config.json',
          model / 'gemma4.vision.gturbo/manifest.json', model / 'mac-app-settings.json']
metadata=[]
for p in files:
    if p.is_file():
        b=p.read_bytes()
        metadata.append(dict(path=str(p), bytes=len(b), sha256=hashlib.sha256(b).hexdigest(), mtime_ns=p.stat().st_mtime_ns))
        if p.name in ('manifest.json','config.json','mac-app-settings.json'):
            (OUT / (p.parent.name+'-'+p.name)).write_bytes(b)
    else:
        metadata.append(dict(path=str(p), absent=True))
(OUT / 'binary-model-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
(OUT / 'commands.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(dict(output=str(OUT), source_file_count=len(hashes), metadata_file_count=len(metadata),
                     tracked_diff_sha256=hashlib.sha256(diff).hexdigest(),
                     commands=[dict(name=r['name'],exit_code=r['exit_code']) for r in records]),indent=2))
