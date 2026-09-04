#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: check-notes.sh <html-file>" >&2
    exit 2
fi

page=$1

if [ ! -f "$page" ]; then
    echo "FAIL: file not found: $page" >&2
    exit 1
fi

fail=0

check_absent() {
    pattern=$1
    message=$2
    if grep -Eq "$pattern" "$page"; then
        echo "FAIL: $message" >&2
        fail=1
    fi
}

check_present() {
    pattern=$1
    message=$2
    if ! grep -Eq "$pattern" "$page"; then
        echo "FAIL: $message" >&2
        fail=1
    fi
}

check_absent '\{\{[^}]+\}\}' "template placeholders remain"
check_absent 'FILL:' "authoring markers remain"
check_absent 'AUTHOR:' "authoring comments remain"
check_absent '<(script|link)[^>]+https?://' "remote script or stylesheet dependency found"
check_present '<meta name="viewport"' "viewport metadata is missing"
check_present '<!doctype html>' "HTML5 doctype is missing"
check_present '<title>[^<]+</title>' "document title is missing"
check_present 'class="promptcard"' "scope card is missing"
check_present 'class="timeline"' "timeline is missing"
check_present 'class="next"' "decision block is missing"
check_present '<script>' "local interaction script is missing"
check_present '</body>' "closing body tag is missing"
check_present '</html>' "closing html tag is missing"

if [ "$fail" -ne 0 ]; then
    exit 1
fi

echo "PASS: $page"
