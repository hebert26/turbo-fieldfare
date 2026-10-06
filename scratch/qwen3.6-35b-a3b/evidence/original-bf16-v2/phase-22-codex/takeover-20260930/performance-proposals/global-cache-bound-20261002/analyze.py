#!/usr/bin/env python3
"""Saved-route cache replay only. Opens JSON inputs, never model payloads."""
import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path

PAIR_BYTES = 6 * 1024 * 1024
REQUEST_SHA = '841f885f3e7e30c56656176e2cb41b4616f01d88db16f30f03ab058f125e1285'
KEYS = [(position, layer) for position in range(1175, 1203) for layer in range(40)]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError('nonfinite JSON: ' + value)

    return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=invalid)


def validate(trace, request):
    require(trace['schema'] == 1 and trace['mode'] == 'off', 'accepted serial OFF required')
    require(trace['captureSHA256'] == REQUEST_SHA and trace['expectedOutputIDsMatched'] is True,
            'capture identity or exact output check')
    require(trace['outputIDs'] == request['sampledTokenIDs'] and len(trace['outputIDs']) == 29,
            'all29 exact output IDs required')
    require(len(request['inputTokenIDs']) == 1175 and request['maximumContext'] == 8192,
            'fixed input geometry')
    safety = trace['safety']
    require(safety['captureComplete'] is True and safety['runnerUsable'] is True
            and safety['position'] == 1203 and safety['pendingTokens'] == 1, 'complete trace required')
    require(trace['sourceValidation']['failures'] == 0 and trace['minimumFreePercent'] >= 30,
            'successful OFF safety evidence required')
    plans, routes = trace['plans'], trace['routes']
    require(len(plans) == len(routes) == 1120, '1120 maps required')
    require([(row['position'], row['layer']) for row in plans] == KEYS, 'plan order/coverage')
    require([(row['position'], row['layer']) for row in routes] == KEYS, 'route order/coverage')
    initial, current = {}, {}
    transitions, layer_counts = [], [{'misses': 0, 'hits': 0} for _ in range(40)]
    groups = []
    for index, (plan, route) in enumerate(zip(plans, routes)):
        layer = plan['layer']
        experts, slots, misses = plan['experts'], plan['assignedSlots'], plan['missIndices']
        residents = plan['residents']
        require(experts == route['experts'] and len(experts) == len(set(experts)) == 8,
                'route/plan Top8 mismatch at ' + str(index))
        require(all(type(expert) is int and 0 <= expert < 256 for expert in experts), 'expert range')
        require(len(route['weightBits']) == 8 and all(type(v) is int and 0 <= v < 2**32
                for v in route['weightBits']), 'weight bit shape')
        require(len(residents) == 16 and all(v is None or type(v) is int and 0 <= v < 256
                for v in residents), '16-slot resident shape')
        live = [v for v in residents if v is not None]
        require(len(live) == len(set(live)), 'duplicate resident')
        if layer not in initial:
            initial[layer] = list(residents)
            current[layer] = list(residents)
        require(current[layer] == residents, 'recorded slot transition mismatch at ' + str(index))
        require(len(slots) == len(set(slots)) == 8 and all(type(v) is int and 0 <= v < 16
                for v in slots), 'assigned slot shape')
        require(len(misses) == len(set(misses)) and all(type(v) is int and 0 <= v < 8
                for v in misses), 'miss index shape')
        expected_misses = [member for member, expert in enumerate(experts) if expert not in live]
        require(misses == expected_misses, 'miss membership/order mismatch at ' + str(index))
        before = list(current[layer])
        for member, (expert, slot) in enumerate(zip(experts, slots)):
            if member not in misses:
                require(before[slot] == expert, 'hit slot mismatch at ' + str(index))
            else:
                current[layer][slot] = expert
        require(set(experts) <= set(current[layer]), 'current Top8 not held together')
        require(len(set(current[layer])) == 16, 'published cache duplicate or empty slot')
        layer_counts[layer]['misses'] += len(misses)
        layer_counts[layer]['hits'] += 8 - len(misses)
        groups.append(tuple((layer, expert) for expert in experts))
        transitions.append({'index': index, 'position': plan['position'], 'layer': layer,
                            'before': before, 'after': list(current[layer]),
                            'experts': experts, 'assignedSlots': slots, 'missIndices': misses})
    require(len(initial) == 40 and all(len(set(v)) == 16 and None not in v for v in initial.values()),
            'complete40x16 initial inventory required')
    require(sum(row['misses'] for row in layer_counts) == 5337
            and sum(row['hits'] for row in layer_counts) == 3623, 'recorded LFU totals differ')
    return initial, groups, layer_counts, transitions


def simulate(initial, groups, global_storage):
    # Future positions choose victims only. Admissions always wait for demand.
    future = defaultdict(deque)
    for index, group in enumerate(groups):
        for pair in group:
            future[pair].append(index)
    pools = {0: set((layer, expert) for layer, values in initial.items() for expert in values)} \
        if global_storage else {layer: set((layer, expert) for expert in values)
                               for layer, values in initial.items()}
    capacity = 640 if global_storage else 16
    counts = [{'misses': 0, 'hits': 0} for _ in range(40)]
    records, compulsory = [], [0] * 40
    seen = set(pair for pool in pools.values() for pair in pool)
    for index, group in enumerate(groups):
        layer = group[0][0]
        required = set(group)
        pool = pools[0 if global_storage else layer]
        require(len(required) == 8 and len(pool) == capacity, 'capacity or grouped pin invariant')
        for pair in group:
            require(future[pair].popleft() == index, 'future queue alignment')
        missing = required - pool
        for pair in group:
            if pair not in seen:
                compulsory[layer] += 1
                seen.add(pair)
        candidates = pool - required
        require(len(candidates) >= len(missing), 'insufficient unpinned victims')
        # Never evict any current consumer. Largest next-use wins; ties use identity.
        victims = sorted(candidates,
                         key=lambda pair: (future[pair][0] if future[pair] else len(groups), pair),
                         reverse=True)[:len(missing)]
        pool.difference_update(victims)
        pool.update(missing)
        require(required <= pool and len(pool) == capacity, 'whole Top8/capacity after admission')
        counts[layer]['misses'] += len(missing)
        counts[layer]['hits'] += 8 - len(missing)
        records.append({'index': index, 'position': KEYS[index][0], 'layer': layer,
                        'missExperts': sorted(pair[1] for pair in missing),
                        'victims': [list(pair) for pair in victims], 'allEightPinned': True})
    misses = sum(row['misses'] for row in counts)
    require(misses >= sum(compulsory) and misses + sum(row['hits'] for row in counts) == 8960,
            'compulsory or total request accounting')
    return {'misses': misses, 'hits': 8960 - misses, 'logicalReadBytes': misses * PAIR_BYTES,
            'compulsoryFirstTouches': sum(compulsory), 'perLayer': counts,
            'compulsoryPerLayer': compulsory, 'maps': records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(sha(args.request) == REQUEST_SHA, 'request hash differs')
    trace, request = read(args.trace), read(args.request)
    initial, groups, actual, transitions = validate(trace, request)
    local = simulate(initial, groups, False)
    shared = simulate(initial, groups, True)
    require(local['compulsoryFirstTouches'] == shared['compulsoryFirstTouches'], 'first-touch mismatch')
    saved = local['misses'] - shared['misses']
    passed = shared['misses'] * 4 <= local['misses'] * 3
    per_layer = [{'layer': layer, 'recordedLFU': actual[layer],
                  'localFutureAware': local['perLayer'][layer], 'globalFutureAware': shared['perLayer'][layer],
                  'globalMissesSavedVsLocal': local['perLayer'][layer]['misses'] - shared['perLayer'][layer]['misses'],
                  'compulsoryFirstTouches': local['compulsoryPerLayer'][layer]} for layer in range(40)]
    result = {'schema': 1, 'completed': True, 'tracePath': str(args.trace.resolve()),
              'traceSHA256': sha(args.trace), 'requestPath': str(args.request.resolve()),
              'requestSHA256': sha(args.request), 'scriptSHA256': sha(Path(__file__)),
              'maps': 1120, 'forwardGroups': 28, 'consumers': 8960,
              'capacityPairs': 640, 'capacityBytes': 640 * PAIR_BYTES, 'pairBytes': PAIR_BYTES,
              'initialResidentPairs': 640, 'allCurrentTop8Pinned': True,
              'recordedLFU': {'misses': 5337, 'hits': 3623, 'logicalReadBytes': 5337 * PAIR_BYTES},
              'localFutureAware': {k: v for k, v in local.items() if k not in ('maps', 'perLayer', 'compulsoryPerLayer')},
              'globalFutureAware': {k: v for k, v in shared.items() if k not in ('maps', 'perLayer', 'compulsoryPerLayer')},
              'globalMissesSavedVsLocal': saved, 'globalMissReductionVsLocal': saved / local['misses'],
              'logicalBytesSavedVsLocal': saved * PAIR_BYTES, 'perLayer': per_layer,
              'gate': {'rule': 'global misses <=75% of local misses', 'passed': passed,
                       'decision': 'supports separate causal cache design review' if passed else 'reject this fixed shared-cache screen'},
              'algorithm': 'Demand-only admission, equal-sized pair identity(layer,expert), complete currentTop8 pinned. Evict farthest next requested unpinned resident; no-future-use ranks last. Fixed identity tie break. Same method and initial bytes in both layouts.',
              'optimality': 'These are achievable clairvoyant simulation counts, hence upper bounds on minimum misses under their respective storage constraints. Grouped-pin optimality is not proved. They are not certified lower bounds or causal policy results.',
              'limits': ['Future routes are available only to offline victim selection. No future preload, free rewarm or cache reset.',
                         'Initial global inventory comes from the first pre-plan snapshot of each private layer. Serial execution has no writer to that layer before its first map.',
                         'Recorded slot transitions are reconstructed exactly. Initial LFU use counts/last-use clocks are absent, so its victim ranking is not independently recomputed.',
                         'Initial prefill-resident pages are not counted as compulsory first touches. A first request for any other pair is compulsory in every demand-only policy.',
                         'No physical I/O, source checks, GPU leases, memory pressure, residency transfer or runtime latency is simulated. Logical bytes are not SSD bytes.',
                         'A failed greedy clairvoyant comparison is the assigned preliminary screen, not a proof that all global policies or architectures fail. No runtime speed claim.']}
    args.output.mkdir(exist_ok=False)
    write(args.output / 'result.json', result)
    write(args.output / 'actual-transitions.json', transitions)
    write(args.output / 'initial-residents.json', {str(layer): initial[layer] for layer in range(40)})
    write(args.output / 'local-maps.json', local['maps'])
    write(args.output / 'global-maps.json', shared['maps'])
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
