#!/usr/bin/env python3
"""Read saved memory evidence. Check serial output only after completion."""
import argparse
import datetime
import json
import pathlib
import re
import uuid

from evidence_helpers import BASE, sha
from run import REQUEST_SHA256, require, write_json

SOURCE_KEYS = ('tokenAdmissions', 'layerAdmissions', 'reusedPairChecks',
               'preGPUChecks', 'sharedChecks', 'finalChecks')
CPU_KEYS = ('processCPUUserBeforeMicroseconds', 'processCPUSystemBeforeMicroseconds',
            'processCPUUserAfterMicroseconds', 'processCPUSystemAfterMicroseconds')


def parse_json(text):
    def reject(value):
        raise ValueError('non-finite JSON value: ' + value)

    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, 'duplicate JSON key: ' + key)
            value[key] = item
        return value

    return json.loads(text, parse_constant=reject, object_pairs_hook=unique)


def load_json(path):
    return parse_json(path.read_text())


def integer(value, label, minimum=0, maximum=2**64 - 1):
    require(type(value) is int and minimum <= value <= maximum, 'invalid integer: ' + label)


def token_list(value, label):
    require(isinstance(value, list), 'invalid token list: ' + label)
    for token in value:
        integer(token, label, maximum=2**31 - 1)


def digest(value, label):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
            'invalid SHA256: ' + label)


def check_io(value):
    require(isinstance(value, dict), 'missing read counters')
    for key in ('reads', 'failedReads'):
        require(isinstance(value[key], list) and len(value[key]) == 4, 'invalid IO array: ' + key)
        for number in value[key]:
            integer(number, 'IO ' + key)
    require(all(number == 0 for number in value['failedReads']), 'failed reads')
    for key, length, fields in [('preads', 4, ('count', 'bytes', 'errors')),
                                 ('validations', 16, ('count', 'errors'))]:
        require(isinstance(value[key], list) and len(value[key]) == length, 'invalid IO array: ' + key)
        for counter in value[key]:
            require(isinstance(counter, dict), 'invalid IO counter')
            for field in fields:
                integer(counter[field], 'IO ' + key + ' ' + field)
            require(counter['errors'] == 0, 'IO counter errors')
    require(value['reads'][:2] == [0, 0] and value['failedReads'][:2] == [0, 0],
            'cycle contains prefill reads')
    require(all(counter['count'] == 0 for counter in value['validations'][:8])
            and all(counter['count'] == 0 for counter in value['preads'][:2]),
            'cycle contains prefill counters')


def analyze_report(result, request):
    integer(result['schema'], 'schema', minimum=1, maximum=1)
    require(result['mode'] == 'serial', 'serial-only probe required')
    require(request['maximumContext'] == 8192 and len(request['inputTokenIDs']) == 1175,
            'fixed prompt or context differs')
    require(request['settings']['temperature'] == 0, 'saved sampling temperature differs')
    require(result['requestSHA256'] == REQUEST_SHA256, 'actual request digest differs')
    require(result['sourceDescriptorSHA256'] == request['sourceDescriptorSHA256'], 'actual source descriptor differs')
    require(result['prefixTokens'] == 1175 and result['maximumContext'] == 8192
            and result['maximumOutputs'] == 29 and result['expertSlots'] == 16
            and result['expertCachePolicy'] == 'lfu' and result['diagnosticOnly'] is True,
            'fixed causal workload settings differ')
    integer(result['acceptedRouteCount'], 'accepted routes', minimum=1120, maximum=1120)
    integer(result['acceptedRawRowCount'], 'accepted raw rows', minimum=28, maximum=28)
    digest(result['acceptedRouteSHA256'], 'accepted routes')
    digest(result['acceptedRawRowSHA256'], 'accepted raw rows')
    expected = request['sampledTokenIDs']
    token_list(expected, 'saved output IDs')
    require(len(expected) == 29 and expected[-1] == request['codecEOS'], 'saved EOS output differs')
    token_list(result['outputIDs'], 'actual output IDs')
    require(result['outputIDs'] == expected and result['expectedOutputIDsMatched'] is True,
            'actual 29 output IDs, including EOS, differ')
    integer(result['promptWallNanoseconds'], 'prompt wall', minimum=1)
    integer(result['decodeWallNanoseconds'], 'whole decode wall', minimum=1)
    integer(result['requestWallNanoseconds'], 'whole request wall', minimum=1)
    require(result['requestWallNanoseconds'] >= result['promptWallNanoseconds']
            + result['decodeWallNanoseconds'], 'whole request excludes a measured phase')
    integer(result['minimumFreePercent'], 'internal free memory', minimum=30, maximum=100)
    state = result['finalState']
    require(isinstance(state, dict) and state['runnerUsable'] is True, 'final runner unusable')
    integer(state['position'], 'final position')
    integer(state['pendingTokens'], 'pending token count')
    require(state['position'] == 1203 and state['pendingTokens'] == 1, 'final consumed or pending state differs')
    layers = []
    for key, length in [('linearPositions', 30), ('fullKVPositions', 10)]:
        positions = state[key]
        require(isinstance(positions, dict) and len(positions) == length, 'final layer map differs: ' + key)
        for layer, position in positions.items():
            require(re.fullmatch(r'(?:0|[1-9][0-9]*)', layer) is not None, 'invalid layer key')
            integer(position, 'final layer position', minimum=1203, maximum=1203)
            layers.append(int(layer))
    require(sorted(layers) == list(range(40)), 'final layer maps overlap or miss a layer')
    for key in ('linearStateSHA256', 'committedKeysSHA256', 'committedValuesSHA256'):
        digest(state[key], key)
    for key in CPU_KEYS:
        integer(result[key], key)
    require(result[CPU_KEYS[2]] >= result[CPU_KEYS[0]]
            and result[CPU_KEYS[3]] >= result[CPU_KEYS[1]], 'CPU counters moved backward')
    cycles = result['cycles']
    require(isinstance(cycles, list) and 1 <= len(cycles) <= 28, 'invalid causal cycle count')
    consumed = 0
    emitted = [result['outputIDs'][0]]
    summary = {'cycleCalls': len(cycles), 'serialCycles': 0, 'pairCycles': 0, 'shortProposalSerialCycles': 0, 'branchCycles': 0,
               'rejectedPairCycles': 0, 'rejectedBranchCycles': 0, 'verifiedInputs': 0, 'consumedInputs': 0,
               'proposedDrafts': 0, 'acceptedDrafts': 0, 'rejectedVerifiedInputs': 0,
               'replayedInputs': 0, 'recurrentAppendCalls': 0, 'recurrentKeptInputs': 0,
               'fullReplayWallNanoseconds': 0, 'recurrentOnlyWallNanoseconds': 0,
               'maximumSavedInputBytes': 0, 'recurrentOnlyKeepCounts': {}, 'branchSamplerCalls': 0, 'prefillSamplerCalls': 1,
               'reportedCacheHits': 0, 'reportedCacheMisses': 0,
               'cycleWallNanoseconds': 0, 'settlementWallNanoseconds': 0,
               'preadCalls': 0, 'preadReturnedBytes': 0, 'validationCalls': 0,
               'sourceChecks': {key: 0 for key in SOURCE_KEYS}}
    for index, cycle in enumerate(cycles):
        require(isinstance(cycle, dict), 'invalid cycle')
        for key in ('index', 'position', 'pendingInput', 'matchedDrafts', 'consumedInputs',
                    'verifiedInputs', 'wallNanoseconds', 'restoreReplayNanoseconds', 'hits', 'misses'):
            integer(cycle[key], 'cycle ' + key)
        require(cycle['index'] == index and cycle['position'] == 1175 + consumed,
                'cycle index or consumed position differs')
        require(cycle['pendingInput'] == emitted[-1] and emitted[-1] != request['codecEOS'],
                'cycle pending input differs or consumes terminal EOS')
        proposals, actual = cycle['proposalIDs'], cycle['provisionalIDs']
        token_list(proposals, 'causal proposals')
        token_list(actual, 'actual branch samples')
        matched, used, verified = cycle['matchedDrafts'], cycle['consumedInputs'], cycle['verifiedInputs']
        require(len(proposals) <= 3 and 1 <= verified <= 4 and verified == 1 + len(proposals),
                'cycle exceeds fixed max4 or branch row count differs')
        require(matched <= len(proposals) and used == 1 + matched and used <= verified,
                'accepted or consumed count differs')
        require(len(actual) == used and actual[:matched] == proposals[:matched],
                'actual branch samples do not match accepted prefix')
        if matched < len(proposals) and actual[-1] != request['codecEOS']:
            require(actual[-1] != proposals[matched], 'reported rejection has no mismatch')
        require(request['codecEOS'] not in actual[:-1], 'sampling continued after EOS')
        require(cycle['wallNanoseconds'] > 0
                and cycle['restoreReplayNanoseconds'] <= cycle['wallNanoseconds'], 'cycle timing differs')
        if result['mode'] == 'serial':
            require(verified == used == 1 and not proposals and matched == 0
                    and cycle['restoreReplayNanoseconds'] == 0, 'serial mode used branch or replay work')
        for key in ('recurrentAppendCalls', 'recurrentKeptInputs', 'normalizedInputBytes'):
            integer(cycle[key], 'cycle ' + key)
        kind = cycle['recoveryKind']
        rejected = verified > used
        expected_kind = ('recurrentOnly' if result['mode'] == 'new' and verified == 4
                         else 'fullReplay') if rejected else 'none'
        require(kind == expected_kind, 'recovery path differs from fixed policy')
        expected_bytes = 983040 if result['mode'] == 'new' and verified == 4 else 0
        require(cycle['normalizedInputBytes'] == expected_bytes, 'saved normalized row bytes differ')
        summary['maximumSavedInputBytes'] = max(summary['maximumSavedInputBytes'], expected_bytes)
        if rejected:
            require(result['mode'] != 'serial' and cycle['restoreReplayNanoseconds'] > 0,
                    'rejected branch lacks settlement evidence')
            summary['rejectedBranchCycles'] += 1
            if verified == 4:
                summary['rejectedPairCycles'] += 1
        else:
            require(cycle['restoreReplayNanoseconds'] == 0, 'unrejected branch reports settlement')
        if kind == 'recurrentOnly':
            require(cycle['recurrentAppendCalls'] == 30 * used
                    and cycle['recurrentKeptInputs'] == used, 'recurrent settlement work differs')
            summary['recurrentAppendCalls'] += cycle['recurrentAppendCalls']
            summary['recurrentKeptInputs'] += used
            summary['recurrentOnlyWallNanoseconds'] += cycle['restoreReplayNanoseconds']
            keep_key = str(used)
            summary['recurrentOnlyKeepCounts'][keep_key] = summary['recurrentOnlyKeepCounts'].get(keep_key, 0) + 1
        else:
            require(cycle['recurrentAppendCalls'] == 0 and cycle['recurrentKeptInputs'] == 0,
                    'nonrecurrent path reports recurrent settlement')
            if kind == 'fullReplay':
                summary['replayedInputs'] += used
                summary['fullReplayWallNanoseconds'] += cycle['restoreReplayNanoseconds']
        digest(cycle['routeSHA256'], 'cycle routes')
        digest(cycle['rawRowSHA256'], 'cycle raw rows')
        checks = cycle['sourceChecks']
        require(isinstance(checks, dict) and set(checks) == set(SOURCE_KEYS), 'source check fields differ')
        for key in SOURCE_KEYS:
            integer(checks[key], 'source checks ' + key)
            summary['sourceChecks'][key] += checks[key]
        io = cycle['io']
        check_io(io)
        summary['preadCalls'] += sum(counter['count'] for counter in io['preads'])
        summary['preadReturnedBytes'] += sum(counter['bytes'] for counter in io['preads'])
        summary['validationCalls'] += sum(counter['count'] for counter in io['validations'])
        summary['serialCycles' if verified == 1 else 'pairCycles' if verified == 4
                else 'shortProposalSerialCycles'] += 1
        if verified > 1:
            summary['branchCycles'] += 1
        for total, value in [('verifiedInputs', verified), ('consumedInputs', used),
                             ('proposedDrafts', len(proposals)), ('acceptedDrafts', matched),
                             ('rejectedVerifiedInputs', verified - used), ('branchSamplerCalls', len(actual)),
                             ('reportedCacheHits', cycle['hits']), ('reportedCacheMisses', cycle['misses']),
                             ('cycleWallNanoseconds', cycle['wallNanoseconds']),
                             ('settlementWallNanoseconds', cycle['restoreReplayNanoseconds'])]:
            summary[total] += value
        consumed += used
        emitted.extend(actual)
    require(consumed == 28 and emitted == result['outputIDs'], 'actual cycle accounting differs from committed output')
    require(summary['cycleWallNanoseconds'] <= result['decodeWallNanoseconds'], 'cycle walls exceed whole decode')
    transactions = result['transactions']
    require(isinstance(transactions, dict), 'missing transaction counts')
    for key in ('begins', 'finishes', 'restores'):
        integer(transactions[key], 'transactions ' + key)
    expected_turns = 1 if result['mode'] == 'serial' else len(cycles)
    require(transactions['begins'] == transactions['finishes'] == expected_turns
            and transactions['restores'] == summary['rejectedBranchCycles'], 'transaction accounting differs')
    settlement = result['settlement']
    require(isinstance(settlement, dict), 'missing settlement totals')
    for key in ('wallNanoseconds', 'rejectedCycles', 'replayedInputs', 'recurrentAppendCalls',
                'recurrentKeptInputs', 'maximumSavedInputBytes', 'liveSavedInputBytesAtEnd'):
        integer(settlement[key], 'settlement ' + key)
    for field, total in [('wallNanoseconds', 'settlementWallNanoseconds'),
                         ('rejectedCycles', 'rejectedBranchCycles'), ('replayedInputs', 'replayedInputs'),
                         ('recurrentAppendCalls', 'recurrentAppendCalls'),
                         ('recurrentKeptInputs', 'recurrentKeptInputs'),
                         ('maximumSavedInputBytes', 'maximumSavedInputBytes')]:
        require(settlement[field] == summary[total], 'settlement aggregate differs: ' + field)
    require(settlement['liveSavedInputBytesAtEnd'] == 0, 'saved recovery rows remain live')
    settled_inputs = summary['replayedInputs'] + summary['recurrentKeptInputs']
    summary.update(settlement=settlement, settlementKeptInputs=settled_inputs,
                   settlementMilliseconds=summary['settlementWallNanoseconds'] / 1e6,
                   settlementMillisecondsPerKeptInput=(
                       summary['settlementWallNanoseconds'] / 1e6 / settled_inputs if settled_inputs else None),
                   requestWallNanoseconds=result['requestWallNanoseconds'],
                   acceptedRouteCount=result['acceptedRouteCount'],
                   acceptedRawRowCount=result['acceptedRawRowCount'],
                   acceptedRouteSHA256=result['acceptedRouteSHA256'],
                   acceptedRawRowSHA256=result['acceptedRawRowSHA256'],
                   transactions=transactions, timingLimits=result['timingLimits'],
                   processCPUCountersMicroseconds={key: result[key] for key in CPU_KEYS},
                   outputCountIncludingEOS=len(emitted), samplerCallsIncludingPrefill=1 + summary['branchSamplerCalls'],
                   terminalEOS=emitted[-1], terminalEOSConsumed=False, finalState=state,
                   promptWallNanoseconds=result['promptWallNanoseconds'],
                   decodeWallNanoseconds=result['decodeWallNanoseconds'],
                   decodeOutsideCyclesNanoseconds=result['decodeWallNanoseconds'] - summary['cycleWallNanoseconds'],
                   processCPUUserDuringDecodeMicroseconds=result[CPU_KEYS[2]] - result[CPU_KEYS[0]],
                   processCPUSystemDuringDecodeMicroseconds=result[CPU_KEYS[3]] - result[CPU_KEYS[1]],
                   targetInputsIncludingReplay=summary['verifiedInputs'] + summary['replayedInputs'])
    return summary


PHASES = {'beforeModelLoad', 'afterModelLoad', 'runnerReady', 'prefillEmbeddingDone',
          'prefillLayerDone', 'residencyRequestBefore', 'residencyRequestAfter',
          'prefillDone', 'decodeDone', 'diagnosticsStart', 'diagnosticsEnd'}
PHASE_NUMBERS = ('currentAllocatedSizeBytes', 'cachedLayerCount', 'allocatedCacheBytes',
                 'residencyRegisteredLayers', 'residencyAllocationCount',
                 'residencyAllocatedSizeBytes')


def read_lines(path, errors, phase_only=False, notes=None):
    if not path.is_file():
        errors.append('Missing file: ' + path.name)
        return []
    rows = []
    text = path.read_text(errors='replace')
    lines = text.splitlines()
    for index, line in enumerate(lines, 1):
        if phase_only and not line.lstrip().startswith('{'):
            continue
        try:
            row = parse_json(line)
            require(isinstance(row, dict), 'record is not an object')
            if not phase_only or row.get('type') == 'memoryPhase':
                rows.append(row)
        except Exception as error:
            if phase_only and isinstance(error, json.JSONDecodeError) and index == len(lines) and not text.endswith('\n'):
                if notes is not None:
                    notes.append('Final stderr JSON line is incomplete. No record was made from it.')
            else:
                errors.append(path.name + ' line ' + str(index) + ': ' + str(error))
    return rows


def sample_change(rows, field):
    values = [(row['startUptimeNanoseconds'], row['values'].get(field)) for row in rows
              if type(row['values'].get(field)) is int]
    if not values:
        return None
    return {'first': values[0][1], 'last': values[-1][1],
            'peakSampled': max(value for _, value in values),
            'change': values[-1][1] - values[0][1],
            'firstUptimeNanoseconds': values[0][0], 'lastUptimeNanoseconds': values[-1][0]}


def memory_summary(run, receipt):
    errors = []
    notes = []
    phases = []
    for row in read_lines(run / 'probe.stderr', errors, phase_only=True, notes=notes):
        try:
            require(row['schema'] == 1 and row['pid'] == receipt['ownedPID'], 'phase identity differs')
            require(row['phase'] in PHASES, 'unknown phase')
            integer(row['uptimeNanoseconds'], 'phase time', minimum=1)
            require(isinstance(row['requestID'], str) and row['requestID'], 'missing request ID')
            if phases:
                require(row['requestID'] == phases[0]['requestID']
                        and row['uptimeNanoseconds'] >= phases[-1]['uptimeNanoseconds'],
                        'phase identity or time changed')
            for key in PHASE_NUMBERS:
                if row.get(key) is not None:
                    integer(row[key], key)
            require(row.get('layer') is None or type(row['layer']) is int
                    and 0 <= row['layer'] < 40, 'invalid phase layer')
            require(row.get('residencyRequested') is None
                    or type(row['residencyRequested']) is bool, 'invalid residency flag')
            phases.append(row)
        except Exception as error:
            errors.append('Phase record: ' + str(error))
    phase_errors = list(errors)
    sampler_rows = read_lines(run / 'process-system.jsonl', errors)
    process, system = [], []
    for row in sampler_rows:
        if row.get('kind') == 'configuration':
            for key in ('clockError', 'processError'):
                if row.get(key):
                    errors.append(key + ': ' + row[key])
        if row.get('kind') != 'sample':
            continue
        usage = row.get('process', {})
        if row.get('pid') != receipt.get('ownedPID') or usage.get('identityChanged'):
            errors.append('Sampler identity differs')
            continue
        if usage.get('returnCode') == 0 and isinstance(usage.get('values'), dict):
            process.append(usage)
        else:
            errors.append('Process sample unavailable: ' + str(usage.get('error', usage.get('returnCode'))))
        vm = row.get('systemVM', {})
        if vm.get('exit') == 0 and isinstance(vm.get('counts'), dict) and usage.get('returnCode') == 0:
            system.append(vm)
        elif vm.get('exit') != 0 or not isinstance(vm.get('counts'), dict):
            errors.append('System sample unavailable: ' + str(vm.get('error', vm.get('exit'))))
    vm_changes = {}
    for field in ('Pages wired down', 'Pages occupied by compressor', 'Pages stored in compressor',
                  'Compressions', 'Decompressions', 'Swapins', 'Swapouts'):
        present = [(row['startUptimeNanoseconds'], row['counts'][field]) for row in system
                   if field in row['counts']]
        vm_changes[field] = None if not present else {
            'first': present[0][1], 'last': present[-1][1],
            'change': present[-1][1] - present[0][1],
            'firstUptimeNanoseconds': present[0][0], 'lastUptimeNanoseconds': present[-1][0]}
    allocated = [row['currentAllocatedSizeBytes'] for row in phases
                 if row.get('currentAllocatedSizeBytes') is not None]
    fields = ('ri_phys_footprint', 'ri_lifetime_max_phys_footprint', 'ri_wired_size',
              'ri_resident_size', 'ri_diskio_bytesread')
    stats = {field: sample_change(process, field) for field in fields}
    missing = [field for field, value in stats.items() if value is None]
    missing += [field for field, value in vm_changes.items() if value is None]
    stop = receipt.get('stopRequestedUptimeNanoseconds', receipt.get('childEndUptimeNanoseconds'))
    before_stop = [row for row in phases if stop is None or row['uptimeNanoseconds'] <= stop]
    return {'phaseRecordCount': len(phases), 'lastDurablePhase': before_stop[-1] if before_stop else None,
            'stopUptimeNanoseconds': stop,
            'peakMetalAllocatedSizeAtMarkersBytes': max(allocated) if allocated else None,
            'processSampleCount': len(process), 'processCounters': stats,
            'systemSampleCount': len(system), 'systemVMChanges': vm_changes,
            'systemPageSizeBytes': system[-1].get('pageSizeBytes') if system else None,
            'missingCounters': missing, 'errors': errors, 'phaseRecordErrors': phase_errors, 'notes': notes,
            'stopReason': receipt.get('watchdogAbort', receipt.get('error')),
            'launchFreePercent': receipt.get('launchMemoryFreePercent'),
            'minimumFreePercent': receipt.get('minimumFreePercent'),
            'childExit': receipt.get('childExit'), 'wallSeconds': receipt.get('wallSeconds')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()
    run = args.run.resolve()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = args.output or run / ('analysis-' + stamp + '-' + uuid.uuid4().hex + '.json')
    verdict = {'schema': 1, 'runPath': str(run), 'exit': 2, 'validEvidence': False,
               'completeCorrectnessResult': False, 'speedClaim': False, 'errors': [],
               'limitations': ['The last marker does not prove the active phase at stop.',
                               'Peaks are sampled values. Missing counters remain missing.',
                               'Metal allocated size, process footprint, and system VM are different measures.',
                               'residencyRequestAfter means the call returned. It does not prove physical residency.',
                               'System changes cannot prove which process caused them.',
                               'Decode timing is exploratory. Request timing excludes model load.']}
    try:
        receipt = load_json(run / 'receipt.json')
        verdict['runReceiptSHA256'] = sha(run / 'receipt.json')
        require(receipt['runPath'] == str(run) and receipt['requestedMode'] == 'serial', 'run identity differs')
        raw = receipt.get('rawSHA256', {})
        require(isinstance(raw, dict) and raw, 'missing raw evidence hashes')
        for name, wanted in raw.items():
            path = pathlib.Path(name)
            require(not path.is_absolute() and '..' not in path.parts, 'unsafe evidence path')
            require(sha(run / path) == wanted, 'raw evidence changed: ' + name)
        require('probe.stderr' in raw and 'memory.jsonl' in raw, 'missing child or guard evidence')
        verdict['memory'] = memory_summary(run, receipt)
        require(receipt.get('integrityVerified') is True, 'source, binary, or control proof did not complete')
        require(not verdict['memory']['phaseRecordErrors'], 'invalid phase records')
        verdict['validEvidence'] = True
        if receipt['exit'] == 0 and receipt.get('childExit') == 0 and not receipt.get('watchdogAbort'):
            require(sha(BASE / 'request.json') == REQUEST_SHA256, 'saved request differs')
            result = load_json(run / 'result.json')
            require(result == load_json(run / 'probe.stdout'), 'stdout and result differ')
            integer(receipt['minimumFreePercent'], 'wrapper free memory', minimum=30, maximum=100)
            integer(receipt['launchMemoryFreePercent'], 'launch memory', minimum=50, maximum=100)
            verdict['correctness'] = analyze_report(result, load_json(BASE / 'request.json'))
            verdict['completeCorrectnessResult'] = True
        verdict['exit'] = 0
    except (Exception, KeyboardInterrupt) as error:
        verdict['errors'].append(str(error))
    write_json(output, verdict)
    print(json.dumps({'receipt': str(output), 'validEvidence': verdict['validEvidence'],
                      'completeCorrectnessResult': verdict['completeCorrectnessResult'],
                      'exit': verdict['exit']}))
    return verdict['exit']


if __name__ == '__main__':
    raise SystemExit(main())
