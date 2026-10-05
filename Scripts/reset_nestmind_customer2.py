#!/usr/bin/env python3
"""Remove only NestMind Debug from the booted Customer 2 simulator."""

import json
import plistlib
import subprocess
from pathlib import Path


UDID = "DE8B571C-2234-498F-9FAC-71C96B614792"
BUNDLE = "com.hebertgo.nestmind.debug"
DEVICE = "NestMind Customer 2"


def simctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["xcrun", "simctl", *args], text=True, capture_output=True, timeout=45
    )


def main() -> None:
    devices = simctl("list", "devices", "booted")
    devices.check_returncode()
    if f"{DEVICE} ({UDID}) (Booted)" not in devices.stdout:
        raise SystemExit("Customer 2 is not booted; nothing was removed.")

    installed = simctl("get_app_container", UDID, BUNDLE, "app")
    installed.check_returncode()
    app = Path(installed.stdout.strip())
    with (app / "Info.plist").open("rb") as file:
        actual_bundle = plistlib.load(file)["CFBundleIdentifier"]
    if actual_bundle != BUNDLE:
        raise SystemExit("Installed bundle does not match NestMind Debug; nothing was removed.")

    removed = simctl("uninstall", UDID, BUNDLE)
    removed.check_returncode()
    if simctl("get_app_container", UDID, BUNDLE, "app").returncode == 0:
        raise SystemExit("NestMind Debug is still installed.")
    print(json.dumps({"device": DEVICE, "udid": UDID, "removed_bundle": BUNDLE}))


if __name__ == "__main__":
    main()
