#!/usr/bin/env python3
"""Run one guarded diagnostic child and save its complete evidence."""
import argparse
import datetime
import json
import os
import pathlib
import re
import subprocess
import time
import uuid

from evidence_helpers import BASE, sha, overlay_hashes

ROOT = next(p for p in BASE.parents if (p / '.git').exists())
HEAD = 'f210c7067b20e5c12422f59d5759d0175f474502'
REQUEST_SHA256 = '841f885f3e7e30c56656176e2cb41b4616f01d88db16f30f03ab058f125e1285'
FLAGS = {
    'TURBO_QWEN_SOURCE_VALIDATION_FAST': '1',
    'TURBO_QWEN_EXPERT_CACHE_RESIDENCY': '1',
    'TURBO_QWEN_GPU_LINEAR_PREPARATION': '1',
    'TURBO_QWEN_GROUPED_LINEAR_PREFILL': '1',
    'TURBO_QWEN_EXPERT_PROJECTION_BATCH': '0',
    'TURBO_QWEN_SOURCE_MEMBERSHIP_SCAN': '0',
    'TURBO_QWEN_EXPERT_READ_KNOWN_NONE_4': '0',
    'TURBO_QWEN_EXACT_TOKEN_CAPTURE': '0',
}
PROCESS_PATTERN = (
    'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|'
    'TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|'
    'mlx_lm|mlx-lm|TurboFieldfareExactMetadataCost|TurboFieldfareExactMetadataParallel|'
    'TurboFieldfareExactBlockProbe|TurboFieldfareExpertPredictRecall|TurboFieldfareExpertPredictCPU|'
    'TurboFieldfareExpertPrefetchOverlap|resident-kernel-timing|source64-token4|qwen.*harness'
)


def require(value, message):
    if not value:
        raise ValueError(message)


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('registration')
    parser.add_argument('--mode', choices=('off', 'on'), required=True)
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    run = BASE / ('run-' + stamp + '-' + uuid.uuid4().hex)
    run.mkdir()
    flags = dict(FLAGS, TURBO_QWEN_EXPERT_PREFETCH_OVERLAP='1' if args.mode == 'on' else '0')
    report = {'schema': 1, 'requestedMode': args.mode, 'runPath': str(run),
              'commands': [], 'environment': flags, 'memorySamplingSeconds': 2,
              'wholeRunTimeoutSeconds': 420, 'minimumFreePercent': None,
              'wrapperSHA256': sha(pathlib.Path(__file__)), 'exit': 2}
    child = None
    lock = BASE / '.run.lock'
    lock_owned = False

    def command(argv, label):
        item = {'argv': argv, 'stdoutPath': str(run / (label + '.stdout')),
                'stderrPath': str(run / (label + '.stderr')), 'exit': None}
        report['commands'].append(item)
        try:
            value = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=120)
            item['exit'] = value.returncode
            stdout, stderr = value.stdout, value.stderr
        except subprocess.TimeoutExpired as error:
            stdout, stderr = error.stdout or b'', error.stderr or b''
            stdout = stdout.decode(errors='replace') if isinstance(stdout, bytes) else stdout
            stderr = stderr.decode(errors='replace') if isinstance(stderr, bytes) else stderr
            item['error'] = 'guard command timeout'
            (run / (label + '.stdout')).write_text(stdout)
            (run / (label + '.stderr')).write_text(stderr)
            raise
        (run / (label + '.stdout')).write_text(stdout)
        (run / (label + '.stderr')).write_text(stderr)
        return value

    def stop_owned(reason):
        report['watchdogAbort'] = reason
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                report['forcedKillOwnChild'] = True

    try:
        with lock.open('x') as stream:
            stream.write(json.dumps({'pid': os.getpid(), 'runPath': str(run)}) + '\n')
        lock_owned = True
        head = command(['git', 'rev-parse', 'HEAD'], 'commit')
        require(head.returncode == 0 and head.stdout.strip() == HEAD, 'pinned commit required')
        report['commit'] = head.stdout.strip()
        frozen = json.loads((BASE / 'frozen-hashes.json').read_text())
        require(isinstance(frozen, dict) and bool(frozen), 'frozen stage inventory required')
        for name, digest in frozen.items():
            path = pathlib.Path(name)
            require(not path.is_absolute() and '..' not in path.parts, 'unsafe frozen path')
            require(sha(BASE / path) == digest, 'stage drift: ' + name)
        build = json.loads((BASE / 'build-receipt.json').read_text())
        require(build['exit'] == 0 and build['overlayStable'] is True, 'successful stable build required')
        require(build['stageHashesSHA256'] == sha(BASE / 'frozen-hashes.json'), 'build/stage mismatch')
        require(build['overlayHashesSHA256'] == sha(BASE / 'overlay-after-build-hashes.json'), 'build/source receipt drift')
        sources = json.loads((BASE / 'overlay-after-build-hashes.json').read_text())
        require(overlay_hashes() == sources, 'overlay/dependency drift')
        binary = BASE / 'build/release/TurboFieldfareExpertPrefetchOverlap'
        require(binary.is_file() and sha(binary) == build['binarySHA256'], 'built binary drift')
        require(sha(BASE / 'request.json') == REQUEST_SHA256, 'frozen request drift')
        report.update(binarySHA256=sha(binary), stageHashes=frozen, sourceHashes=sources,
                      stageHashesSHA256=sha(BASE / 'frozen-hashes.json'),
                      sourceHashesSHA256=sha(BASE / 'overlay-after-build-hashes.json'),
                      buildReceiptSHA256=sha(BASE / 'build-receipt.json'), requestSHA256=REQUEST_SHA256)
        pre = command(['python3', str(BASE / 'preflight.py')], 'preflight')
        require(pre.returncode == 0, 'preflight failed')
        pre_summary = json.loads(pre.stdout)
        require(pre_summary['passed'] is True and pre_summary['exit'] == 0, 'preflight receipt failed')
        pre_path = pathlib.Path(pre_summary['receipt'])
        require(pre_path.parent.resolve() == BASE and pre_path.is_file(), 'invalid preflight receipt path')
        pre_receipt = json.loads(pre_path.read_text())
        require(pre_receipt['passed'] is True and pre_receipt['exit'] == 0, 'preflight receipt failed')
        report.update(preflightReceipt=str(pre_path), preflightReceiptSHA256=sha(pre_path))
        hardware = command(['system_profiler', 'SPHardwareDataType'], 'hardware')
        require(hardware.returncode == 0, 'hardware read failed')
        launch_memory = command(['/usr/bin/memory_pressure', '-Q'], 'launch-memory')
        launch_free = re.search(r'free percentage:\s*(\d+)%', launch_memory.stdout)
        require(launch_memory.returncode == 0 and launch_free is not None
                and 30 <= int(launch_free.group(1)) <= 100, 'launch memory below 30% or unavailable')
        report['launchMemoryFreePercent'] = int(launch_free.group(1))
        busy = command(['pgrep', '-fl', PROCESS_PATTERN], 'extra-process-guard')
        require(busy.returncode == 1 and not busy.stdout.strip(), 'competing model or harness')
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('TURBO_QWEN_', 'TURBOFIELDFARE_'))}
        env.update(flags)
        command_line = [str(binary), str(pathlib.Path(args.registration).resolve()),
                        str(BASE / 'request.json'), str(run / 'result.json')]
        report.update(probeCommand=command_line, registrationPath=command_line[1],
                      stdoutPath=str(run / 'probe.stdout'),
                      stderrPath=str(run / 'probe.stderr'), resultPath=str(run / 'result.json'),
                      memoryPath=str(run / 'memory.jsonl'))
        with (run / 'probe.stdout').open('xb') as out, (run / 'probe.stderr').open('xb') as err, \
                (run / 'memory.jsonl').open('x') as memory:
            start = time.monotonic()
            child = subprocess.Popen(command_line, cwd=ROOT, stdout=out, stderr=err, env=env)
            report['ownedPID'] = child.pid
            next_sample = start
            while child.poll() is None:
                elapsed = time.monotonic() - start
                if elapsed >= 420:
                    stop_owned('whole run 420s deadline')
                    break
                try:
                    sample = subprocess.run(['/usr/bin/memory_pressure', '-Q'], capture_output=True,
                                            text=True, timeout=min(2, 420 - elapsed))
                    match = re.search(r'free percentage:\s*(\d+)%', sample.stdout)
                    free = int(match.group(1)) if match else None
                    record = {'elapsedSeconds': elapsed, 'exit': sample.returncode,
                              'stdout': sample.stdout, 'stderr': sample.stderr, 'freePercent': free}
                    memory.write(json.dumps(record, allow_nan=False) + '\n')
                    memory.flush()
                    if free is not None:
                        previous = report['minimumFreePercent']
                        report['minimumFreePercent'] = free if previous is None else min(previous, free)
                    if sample.returncode != 0 or free is None or free < 30:
                        stop_owned('memory below 30% or unavailable')
                        break
                except subprocess.TimeoutExpired:
                    stop_owned('memory guard timeout')
                    break
                next_sample += 2
                remaining = 420 - (time.monotonic() - start)
                if remaining <= 0:
                    stop_owned('whole run 420s deadline')
                    break
                delay = min(max(0.01, next_sample - time.monotonic()), remaining)
                try:
                    child.wait(timeout=delay)
                except subprocess.TimeoutExpired:
                    pass
            report['childExit'] = child.wait()
            report['exit'] = 2 if 'watchdogAbort' in report else report['childExit']
            report['wallSeconds'] = time.monotonic() - start
        require(report['minimumFreePercent'] is not None, 'no wrapper memory samples')
        require(sha(binary) == report['binarySHA256'], 'binary changed during run')
        require(overlay_hashes() == sources, 'source changed during run')
        for name, digest in frozen.items():
            require(sha(BASE / name) == digest, 'stage changed during run: ' + name)
        if (run / 'result.json').is_file():
            report['resultSHA256'] = sha(run / 'result.json')
        report.update(stdoutSHA256=sha(run / 'probe.stdout'), stderrSHA256=sha(run / 'probe.stderr'),
                      memorySHA256=sha(run / 'memory.jsonl'))
    except (Exception, KeyboardInterrupt) as error:
        stop_owned('wrapper failure: ' + str(error))
        if child is not None:
            report['childExit'] = child.wait()
        report['exit'] = 2
        report['error'] = str(error)
    finally:
        if lock_owned:
            lock.unlink()
        write_json(run / 'receipt.json', report)
        print(json.dumps({'runPath': str(run), 'receipt': str(run / 'receipt.json'),
                          'exit': report['exit']}, allow_nan=False))
    return report['exit']


if __name__ == '__main__':
    raise SystemExit(main())
