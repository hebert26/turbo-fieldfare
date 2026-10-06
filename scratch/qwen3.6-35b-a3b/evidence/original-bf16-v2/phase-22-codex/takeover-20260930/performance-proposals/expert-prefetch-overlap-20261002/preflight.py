#!/usr/bin/env python3
"""Read-only launch preflight. No install, build, model, or app invocation."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import uuid

ROOT = Path('/Users/dev-machine/dev/turbo-fieldfare-personal')
BASE = Path(__file__).resolve().parent
PLAN = BASE / 'preflight-plan.json'
BUILD = BASE.parent / 'exact-token-app-trial-20261002/app-build-019.json'
PACK = ROOT / 'scratch/gemma4.gturbo'
HEAD = 'f210c7067b20e5c12422f59d5759d0175f474502'
BUILD_SHA = 'ad80270396f15832a763f92af037fe233ad6ecd3729ae4795090184f47d35289'
GEMMA_RECEIPT_SHA = '6b085b2172b3005df09af0ee573e466d9e62b0d416d5cd779292b20325958af2'
MANIFEST_SHA = '1cb53c2423f05dfa673e5f0d9a3407aa355b8227b621f8f8e575830a2bb7fa91'
INSTALLED = {
    'TurboFieldfareMac': {
        'path': '/Applications/TurboFieldfare.app/Contents/MacOS/TurboFieldfareMac',
        'sha256': '4985b79913c6cd289f5de7b9ac4c6d284dda404d79a9916a96741a9dcc94fe28'},
    'TurboFieldfareDecodeService': {
        'path': '/Applications/TurboFieldfare.app/Contents/MacOS/TurboFieldfareDecodeService',
        'sha256': '083e80a7155db7993af86d391ef84cf634cc2785b62289d79758c8e04f81a046'}}
PROCESS_PATTERN = ('TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|'
                   'TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|'
                   'mlx_lm|mlx-lm|source64-token4-timing|resident-kernel-timing')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def relative_path(name):
    require(isinstance(name, str), 'non-string receipt path')
    path = Path(name)
    require(name and not path.is_absolute() and '..' not in path.parts, 'unsafe receipt path')
    return path


def regular(path):
    require(stat.S_ISREG(path.stat().st_mode), 'not a regular file: ' + str(path))


def preflight():
    report = {'schema_version': 1, 'preflight_id': str(uuid.uuid4()), 'started_at': now(),
              'required_head': HEAD, 'app_version': 'installed app019 only',
              'checks': [], 'commands': [], 'launch_performed': False,
              'limitations': ['Gemma payloads are stat-checked against the pinned prior verification receipt. '
                              'Only its small manifest is rehashed, not model payloads.',
                              'This snapshot does not lock out processes or later file changes. '
                              'Main must launch immediately after a passing fresh check.']}

    def check(name, operation):
        item = {'name': name, 'started_at': now(), 'passed': False}
        try:
            item['detail'] = operation()
            item['passed'] = True
        except Exception as error:
            item['error'] = str(error)
        item['finished_at'] = now()
        report['checks'].append(item)

    def command(args, expected_exit=0):
        try:
            result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired) as error:
            def text(value):
                return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
            report['commands'].append({'argv': args, 'exit': None, 'error': str(error),
                'stdout': text(getattr(error, 'stdout', '')), 'stderr': text(getattr(error, 'stderr', '')),
                'finished_at': now()})
            raise
        item = {'argv': args, 'exit': result.returncode, 'stdout': result.stdout,
                'stderr': result.stderr, 'finished_at': now()}
        report['commands'].append(item)
        require(result.returncode == expected_exit, 'unexpected command exit: ' + str(args))
        return result.stdout

    def plan():
        regular(PLAN)
        data = json.loads(PLAN.read_text())
        require(data.get('head') == HEAD and data.get('installed') == INSTALLED,
                'plan does not identify the accepted app019/commit')
        return {'path': str(PLAN), 'sha256': digest(PLAN)}

    def clean_source():
        head = command(['git', 'rev-parse', 'HEAD']).strip()
        status = command(['git', 'status', '--porcelain=v1', '--untracked-files=all'])
        require(head == HEAD, 'commit differs from accepted f210c70 checkpoint')
        command(['git', 'diff', '--quiet', '3fe330585beb148cb1954419a0c41cb70ebbd672', HEAD, '--', 'Sources', 'Tests', 'Scripts', 'Package.swift', 'Package.resolved'])
        require(not status.strip(), 'checkout is not clean')
        return {'head': head, 'clean': True}

    def source_hashes():
        regular(BUILD)
        require(digest(BUILD) == BUILD_SHA, 'app-build-019 receipt drift')
        build = json.loads(BUILD.read_text())
        require(build.get('exit') == 0 and build.get('stable') is True
                and build.get('before') == build.get('after'), 'app019 build was not stable/successful')
        expected = build.get('after')
        require(isinstance(expected, dict) and len(expected) == 731, 'unexpected source inventory')
        records = []
        for name, wanted in sorted(expected.items()):
            path = relative_path(name)
            require(path.parts[0] in ('Sources', 'Tests', 'Scripts', 'Package.swift'), 'non-source inventory path')
            current = ROOT / path
            require(current.resolve().is_relative_to(ROOT), 'source path leaves checkout')
            regular(current)
            actual = digest(current)
            records.append({'path': name, 'expected': wanted, 'actual': actual, 'matches': actual == wanted})
        mismatches = [row['path'] for row in records if not row['matches']]
        report['source_hashes'] = records
        require(not mismatches, 'app019 source mismatch: ' + ', '.join(mismatches))
        return {'receipt': str(BUILD), 'receipt_sha256': BUILD_SHA, 'matched': len(records)}

    def installed_app():
        binaries = []
        for name, identity in sorted(INSTALLED.items()):
            path = Path(identity['path'])
            # Never follow an installed symlink into the rejected staging output.
            require(not path.resolve().is_relative_to(ROOT / 'dist'), 'installed binary resolves into rejected dist')
            regular(path)
            actual = digest(path)
            binaries.append({'name': name, 'path': str(path), 'actual': actual,
                             'expected': identity['sha256'], 'matches': actual == identity['sha256']})
        report['installed_binaries'] = binaries
        require(all(row['matches'] for row in binaries), 'installed app is not accepted app019')
        return {'matched': len(binaries), 'dist_read_or_install': False}

    def gemma():
        receipt_path = PACK / 'verified-install.json'
        regular(receipt_path)
        require(digest(receipt_path) == GEMMA_RECEIPT_SHA, 'Gemma verification receipt drift')
        data = json.loads(receipt_path.read_text())
        require(data.get('schemaVersion') == 1 and data.get('toolVersion') == 'TurboFieldfareRepack verify-install',
                'Gemma install is not the previously verified pack')
        require(data.get('manifestSha256') == MANIFEST_SHA, 'Gemma receipt manifest mismatch')
        files = data.get('files')
        require(isinstance(files, dict) and len(files) == 37, 'Gemma receipt file inventory mismatch')
        records = []
        for name, info in sorted(files.items()):
            path = relative_path(name)
            current = PACK / path
            regular(current)
            size = current.stat().st_size  # No model payload open/read/hash.
            expected = info.get('size')
            require(type(expected) is int and expected >= 0, 'invalid receipt size')
            records.append({'path': name, 'expected_size': expected, 'actual_size': size, 'matches': size == expected})
        report['gemma_file_stats'] = records
        require(all(row['matches'] for row in records), 'Gemma verified file stat mismatch')
        actual_manifest = digest(PACK / 'manifest.json')  # Metadata only.
        require(actual_manifest == MANIFEST_SHA, 'Gemma manifest content mismatch')
        return {'receipt_sha256': GEMMA_RECEIPT_SHA, 'manifest_sha256': actual_manifest,
                'stat_matched_files': len(records), 'payload_read': False}

    def macos():
        version = command(['sw_vers', '-productVersion']).strip()
        require(re.fullmatch(r'\d+(?:\.\d+)*', version) is not None and int(version.split('.')[0]) >= 26,
                'macOS26+ required')
        return {'version': version}

    def swift():
        text = command(['xcrun', 'swiftc', '--version'])
        version = re.search(r'Swift version (\d+)\.(\d+)', text)
        require(version is not None and tuple(map(int, version.groups())) >= (6, 2), 'Swift6.2+ required')
        return {'major_minor': list(map(int, version.groups()))}

    def disk():
        free = shutil.disk_usage(ROOT).free
        require(free > 20 * 2**30, 'more than20GiB free disk required')
        return {'free_bytes': free, 'free_GiB': free / 2**30}

    def memory():
        text = command(['memory_pressure', '-Q'])
        value = re.search(r'free percentage:\s*(\d+)%', text)
        require(value is not None and int(value.group(1)) >= 30, 'at least30% system free memory required')
        return {'system_free_percent': int(value.group(1))}

    def processes():
        output = command(['pgrep', '-fl', PROCESS_PATTERN], expected_exit=1)
        require(not output.strip(), 'competing process output')
        return {'matching_processes': [], 'pattern': PROCESS_PATTERN}

    for name, operation in [('pinned_plan', plan), ('clean_commit', clean_source),
                            ('app019_source_hashes', source_hashes), ('installed_app019', installed_app),
                            ('verified_gemma_metadata', gemma), ('macos', macos), ('swift', swift),
                            ('disk', disk), ('memory', memory), ('no_competing_processes', processes)]:
        check(name, operation)
    report['passed'] = all(item['passed'] for item in report['checks'])
    report['exit'] = 0 if report['passed'] else 2
    report['finished_at'] = now()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = BASE / ('preflight-' + stamp + '-' + report['preflight_id'] + '.json')
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except Exception as error:
        print(json.dumps({'passed': False, 'exit': 2, 'error': 'preflight receipt write failed: ' + str(error)}))
        return 2
    print(json.dumps({'passed': report['passed'], 'exit': report['exit'], 'receipt': str(output),
                      'failed_checks': [item['name'] for item in report['checks'] if not item['passed']]}))
    return report['exit']


if __name__ == '__main__':
    raise SystemExit(preflight())
