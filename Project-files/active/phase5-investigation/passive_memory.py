"""Bounded passive samples of the existing app. Never starts a model."""
import datetime
import json
import pathlib
import subprocess

out = pathlib.Path(__file__).resolve().parent / 'passive-memory'
out.mkdir(exist_ok=True)
records=[]
def capture(name, argv):
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    p=subprocess.run(argv, capture_output=True)
    ended=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/(name+'.stdout')).write_bytes(p.stdout)
    (out/(name+'.stderr')).write_bytes(p.stderr)
    records.append(dict(name=name, argv=argv, started_utc=started, ended_utc=ended, exit_code=p.returncode))
    return p

capture('processes', ['pgrep','-fl','TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'])
capture('process-start', ['ps','-p','30726,30791','-o','pid,ppid,lstart,etime,%cpu,rss,vsz,command'])
capture('hardware', ['sysctl','hw.model','hw.memsize','hw.ncpu','machdep.cpu.brand_string'])
capture('pressure-before', ['memory_pressure','-Q'])
capture('vm-before', ['vm_stat'])
capture('swap-before', ['sysctl','vm.swapusage'])
capture('top', ['top','-l','3','-s','10','-pid','30726','-pid','30791','-stats','pid,command,cpu,mem,rprvt,vsize,threads,time'])
capture('vm-after', ['vm_stat'])
capture('swap-after', ['sysctl','vm.swapusage'])
capture('pressure-after', ['memory_pressure','-Q'])
(out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(dict(output=str(out),commands=records),indent=2))
