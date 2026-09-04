#!/bin/bash
# Evaluate code quality and produce scored report
# Usage: bash .codex/skills/eval-and-improve/scripts/evaluate.sh [project-root]

PROJECT_ROOT="${1:-.}"
cd "$PROJECT_ROOT" || exit 1

# Source directory - only scan VisionCapture project code, not dependencies
SRC="./VisionCapture/Sources"
EXCLUDE="build\|SourcePackages\|DerivedData\|Pods\|.build\|Test\|Mock\|Preview"

TOTAL=0
PASSED=0
FAILED=0
VIOLATIONS=""

check() {
    local category="$1"
    local name="$2"
    local cmd="$3"
    local expect_empty="$4"  # "true" = no matches expected, "false" = matches expected

    TOTAL=$((TOTAL + 1))
    local result
    result=$(eval "$cmd" 2>/dev/null)

    if [ "$expect_empty" = "true" ]; then
        if [ -z "$result" ]; then
            PASSED=$((PASSED + 1))
            echo "  ✅ $name"
        else
            FAILED=$((FAILED + 1))
            local count
            count=$(echo "$result" | wc -l | tr -d ' ')
            echo "  ❌ $name ($count violations)"
            VIOLATIONS="$VIOLATIONS\n[$category] $name:\n$result\n"
        fi
    else
        if [ -n "$result" ]; then
            PASSED=$((PASSED + 1))
            echo "  ✅ $name"
        else
            FAILED=$((FAILED + 1))
            echo "  ❌ $name (not found)"
            VIOLATIONS="$VIOLATIONS\n[$category] $name: Expected but not found\n"
        fi
    fi
}

echo "╔══════════════════════════════════════════════════╗"
echo "║         CODE QUALITY EVALUATION                  ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Project: $(basename "$(pwd)")"
echo "Source:  $SRC"
echo "Date:    $(date '+%Y-%m-%d %H:%M')"
echo ""

# ── CRITICAL: Architecture ──
echo "🔴 CRITICAL: Architecture"

check "ARCH" "Single SwiftDataManager owns ModelContainer" \
    "grep -rln 'ModelContainer(' --include='*Manager.swift' $SRC | grep -v 'SwiftDataManager'" \
    "true"

check "ARCH" "Domain managers receive context via init" \
    "grep -rln 'init(context: ModelContext' --include='*Manager.swift' $SRC" \
    "false"

check "ARCH" "No @Environment(\\.modelContext) in Views" \
    "grep -rn '@Environment(\\\\\\.modelContext)' --include='*.swift' $SRC | grep -v 'Manager.swift\|Preview'" \
    "true"

check "ARCH" "No direct context.fetch outside Managers" \
    "grep -rn 'context\.fetch\|context\.insert\|context\.delete' --include='*.swift' $SRC | grep -v 'Manager\.swift\|Test\|Preview\|Mock'" \
    "true"

echo ""

# ── CRITICAL: Swift Concurrency ──
echo "🔴 CRITICAL: Swift Concurrency"

check "CONC" "ViewModels have @MainActor" \
    "grep -rlZ 'class.*ViewModel' --include='*.swift' $SRC | xargs -0 -I{} sh -c 'grep -B3 \"class.*ViewModel\" \"{}\" | grep -q \"@MainActor\" || grep -Hn \"class.*ViewModel\" \"{}\"' | grep -v 'Test\|Mock\|Preview\|Protocol\|//\|extension'" \
    "true"

check "CONC" "No legacy ObservableObject" \
    "grep -rn 'ObservableObject\|@Published\|@StateObject\|@ObservedObject' --include='*.swift' $SRC | grep -v 'Test\|Mock\|Preview\|//'" \
    "true"

check "CONC" "No DispatchQueue.main usage" \
    "grep -rn 'DispatchQueue\.main' --include='*.swift' $SRC | grep -v 'Test\|Mock\|//'" \
    "true"

check "CONC" "Uses .task not .onAppear+Task" \
    "grep -A2 '\.onAppear' --include='*.swift' -rn $SRC | grep 'Task {' | grep -v 'Test\|Mock\|Preview\|//'" \
    "true"

echo ""

# ── CRITICAL: Crash & Security ──
echo "🔴 CRITICAL: Crash & Security"

check "CRASH" "No try! usage" \
    "grep -rn 'try!' --include='*.swift' $SRC | grep -v 'Test\|Mock\|Preview\|//'" \
    "true"

check "CRASH" "No as! usage" \
    "grep -rn ' as!' --include='*.swift' $SRC | grep -v 'Test\|Mock\|Preview\|//'" \
    "true"

check "CRASH" "No [unowned self]" \
    "grep -rn '\[unowned self\]' --include='*.swift' $SRC | grep -v 'Test\|Mock\|//'" \
    "true"

check "CRASH" "No hardcoded secrets" \
    "grep -rEn '\"sk-|\"pk-|\"api_key|\"secret_key|\"password\"' --include='*.swift' $SRC | grep -v 'Test\|Mock\|//\|enum\|case'" \
    "true"

check "CRASH" "No Thread.sleep" \
    "grep -rn 'Thread\.sleep' --include='*.swift' $SRC | grep -v 'Test\|Mock\|//'" \
    "true"

echo ""

# ── WARNING: Memory Leaks ──
echo "🟠 WARNING: Memory Leaks"

check "MEM" "No NSLock/DispatchSemaphore (use actors)" \
    "grep -rn 'NSLock\|DispatchSemaphore' --include='*.swift' $SRC | grep -v 'Test\|Mock\|//'" \
    "true"

echo ""

# ── QUALITY: Logging ──
echo "🟡 QUALITY: Logging"

check "LOG" "No print() in production code" \
    "grep -rn 'print(' --include='*.swift' $SRC | grep -v 'Test\|Mock\|Preview\|#if DEBUG\|// DEBUG\|//.*print'" \
    "true"

check "LOG" "Centralized Logger extension exists" \
    "grep -rln 'extension Logger' --include='*.swift' $SRC" \
    "false"

check "LOG" "No inline Logger(subsystem:) creation" \
    "grep -rn 'Logger(subsystem:' --include='*.swift' $SRC | grep -v 'extension Logger\|Test\|Mock\|logger+extension'" \
    "true"

echo ""

# ── QUALITY: Code Cleanliness ──
echo "🟡 QUALITY: Code Cleanliness"

check "CLEAN" "No empty catch blocks" \
    "grep -rn 'catch { }' --include='*.swift' $SRC | grep -v 'Test\|Mock\|//'" \
    "true"

echo ""

# ── LEGACY: Patterns ──
echo "🟣 LEGACY: Patterns"

check "LEGACY" "No CFRunLoop/performSelector" \
    "grep -rn 'CFRunLoop\|performSelector' --include='*.swift' $SRC | grep -v 'Test\|Mock\|//'" \
    "true"

echo ""

# ── Summary ──
if [ $TOTAL -gt 0 ]; then
    ACCURACY=$(( (PASSED * 100) / TOTAL ))
else
    ACCURACY=0
fi

echo "╔══════════════════════════════════════════════════╗"
echo "║                   SUMMARY                        ║"
echo "╠══════════════════════════════════════════════════╣"
printf "║  Total Checks:  %-30s ║\n" "$TOTAL"
printf "║  Passed:        %-30s ║\n" "$PASSED ✅"
printf "║  Failed:        %-30s ║\n" "$FAILED ❌"
printf "║  Accuracy:      %-30s ║\n" "${ACCURACY}%"
echo "╠══════════════════════════════════════════════════╣"

if [ $ACCURACY -eq 100 ]; then
    echo "║  🎉 100% ACCURACY - ALL CHECKS PASSED!          ║"
else
    echo "║  ⚠️  VIOLATIONS NEED FIXING                      ║"
fi
echo "╚══════════════════════════════════════════════════╝"

if [ -n "$VIOLATIONS" ]; then
    echo ""
    echo "── Violation Details ──"
    echo -e "$VIOLATIONS"
fi

# Exit code: 0 = 100%, 1 = has violations
[ $ACCURACY -eq 100 ]
