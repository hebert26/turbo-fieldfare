#!/usr/bin/env python3
"""Save optional process and system counters. Do not control the model."""
import argparse
import ctypes
import datetime
import json
import pathlib
import re
import subprocess
import time

SDK = pathlib.Path('/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include')
FIELDS = ('ri_user_time', 'ri_system_time', 'ri_pageins', 'ri_wired_size',
          'ri_resident_size', 'ri_phys_footprint', 'ri_proc_start_abstime',
          'ri_proc_exit_abstime', 'ri_diskio_bytesread', 'ri_diskio_byteswritten',
          'ri_lifetime_max_phys_footprint')


class UptimeClock:
    def __init__(self):
        self.error = None
        self.clock_id = None
        self.system = None
        try:
            header = (SDK / '_time.h').read_text()
            self.clock_id = int(re.search(r'_CLOCK_UPTIME_RAW[^=]*=\s*(\d+)', header).group(1))
            self.system = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
            self.system.clock_gettime_nsec_np.argtypes = (ctypes.c_int,)
            self.system.clock_gettime_nsec_np.restype = ctypes.c_uint64
        except Exception as error:
            self.error = str(error)

    def now(self):
        if self.error is not None:
            return None
        return int(self.system.clock_gettime_nsec_np(self.clock_id))


class ProcessCounters:
    def __init__(self):
        self.error = None
        self.layouts = {}
        self.first_identity = None
        try:
            header = (SDK / 'sys/resource.h').read_text()
            classes = {}
            # Read full SDK structs. Do not use guessed field offsets.
            for version in (2, 4):
                body = re.search(r'struct rusage_info_v' + str(version) + r'\s*\{([^}]+)\};', header).group(1)
                fields = []
                for statement in body.split(';'):
                    statement = statement.strip()
                    if not statement:
                        continue
                    match = re.fullmatch(r'uint(8|64)_t\s+(\w+)(?:\[(\d+)\])?', statement)
                    if match is None:
                        raise ValueError('Unsupported SDK field: ' + statement)
                    bits, name, count = match.groups()
                    kind = ctypes.c_uint8 if bits == '8' else ctypes.c_uint64
                    fields.append((name, kind * int(count) if count else kind))
                cls = type('RUsageV' + str(version), (ctypes.Structure,), {'_fields_': fields})
                if ctypes.sizeof(cls) != 16 + 8 * (len(fields) - 1):
                    raise ValueError('SDK layout size differs')
                classes[version] = cls
                self.layouts[str(version)] = {
                    'size': ctypes.sizeof(cls),
                    'fieldOffsets': {name: getattr(cls, name).offset for name, _ in fields}}
            self.cls = classes[4]
            self.lib = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
            self.lib.proc_pid_rusage.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
            self.lib.proc_pid_rusage.restype = ctypes.c_int
        except Exception as error:
            self.error = str(error)

    def read(self, pid, clock):
        row = {'startUptimeNanoseconds': clock.now(), 'endUptimeNanoseconds': None,
               'returnCode': None, 'errno': None, 'uuid': None,
               'values': {name: None for name in FIELDS}}
        if self.error is not None:
            row['error'] = self.error
        else:
            usage = self.cls()
            ctypes.set_errno(0)
            row['returnCode'] = self.lib.proc_pid_rusage(pid, 4, ctypes.byref(usage))
            row['errno'] = ctypes.get_errno()
            if row['returnCode'] == 0:
                row['uuid'] = bytes(usage.ri_uuid).hex()
                row['values'] = {name: int(getattr(usage, name)) if hasattr(usage, name) else None
                                 for name in FIELDS}
                identity = (row['uuid'], row['values']['ri_proc_start_abstime'])
                if self.first_identity is None:
                    self.first_identity = identity
                row['identityChanged'] = self.first_identity != identity
        row['endUptimeNanoseconds'] = clock.now()
        return row


def vm_snapshot(clock):
    row = {'argv': ['/usr/bin/vm_stat'], 'startUptimeNanoseconds': clock.now(),
           'endUptimeNanoseconds': None, 'exit': None, 'pageSizeBytes': None,
           'counts': None, 'stdout': None, 'stderr': None}
    try:
        result = subprocess.run(row['argv'], capture_output=True, text=True, timeout=0.8)
        row.update(exit=result.returncode, stdout=result.stdout, stderr=result.stderr)
        if result.returncode == 0:
            page_size = re.search(r'page size of (\d+) bytes', result.stdout)
            row['pageSizeBytes'] = int(page_size.group(1)) if page_size else None
            counts = {}
            for line in result.stdout.splitlines()[1:]:
                match = re.fullmatch(r'\s*([^:]+):\s*(\d+)\.?\s*', line)
                if match:
                    counts[match.group(1).strip().strip('"')] = int(match.group(2))
            row['counts'] = counts or None
    except Exception as error:
        row['error'] = str(error)
        for key in ('stdout', 'stderr'):
            value = getattr(error, key, None)
            row[key] = value.decode(errors='replace') if isinstance(value, bytes) else value
    row['endUptimeNanoseconds'] = clock.now()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pid', type=int)
    parser.add_argument('output', type=pathlib.Path)
    parser.add_argument('--seconds', type=int, default=420)
    args = parser.parse_args()
    if args.pid <= 0 or not 1 <= args.seconds <= 420:
        raise ValueError('Invalid PID or duration')
    clock, process = UptimeClock(), ProcessCounters()
    start = time.monotonic()
    next_sample = start
    with args.output.open('x') as output:
        def emit(row):
            output.write(json.dumps(row, allow_nan=False) + '\n')
            output.flush()

        emit({'kind': 'configuration', 'schema': 1, 'pid': args.pid,
              'clock': 'SDK CLOCK_UPTIME_RAW', 'clockID': clock.clock_id,
              'clockError': clock.error, 'processError': process.error,
              'layouts': process.layouts, 'intervalSeconds': 2,
              'maximumSeconds': args.seconds,
              'limits': ['Process counters are not Metal allocated size.',
                         'Resident bytes are not physical footprint.',
                         'Read bytes do not measure physical device reads.',
                         'vm_stat is a system snapshot. Missing counters are null.']})
        while time.monotonic() - start < args.seconds:
            usage = process.read(args.pid, clock)
            emit({'kind': 'sample', 'schema': 1, 'pid': args.pid,
                  'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'elapsedSeconds': time.monotonic() - start,
                  'process': usage, 'systemVM': vm_snapshot(clock)})
            if usage.get('identityChanged') or usage['returnCode'] not in (None, 0):
                break
            next_sample += 2
            time.sleep(min(max(0, next_sample - time.monotonic()),
                           max(0, args.seconds - (time.monotonic() - start))))
        emit({'kind': 'finished', 'elapsedSeconds': time.monotonic() - start})


if __name__ == '__main__':
    main()
