#!/bin/bash
# Find code quality violations

echo "=== Violation Check ==="
echo ""
VIOLATIONS=0

# Legacy patterns
echo "🔴 Legacy patterns:"
LEGACY=$(grep -rn "ObservableObject\|@Published\|@StateObject\|@ObservedObject" --include="*.swift" . 2>/dev/null)
if [ -n "$LEGACY" ]; then
    echo "$LEGACY"
    VIOLATIONS=$((VIOLATIONS + $(echo "$LEGACY" | wc -l)))
else
    echo "  ✅ None found"
fi
echo ""

# Empty catches
echo "🔴 Empty catch blocks:"
EMPTY_CATCH=$(grep -rn "catch { }\|catch {\s*$" --include="*.swift" . 2>/dev/null)
if [ -n "$EMPTY_CATCH" ]; then
    echo "$EMPTY_CATCH"
    VIOLATIONS=$((VIOLATIONS + $(echo "$EMPTY_CATCH" | wc -l)))
else
    echo "  ✅ None found"
fi
echo ""

# Print statements (outside DEBUG)
echo "🔴 print() in production code:"
PRINTS=$(grep -rn "print(" --include="*.swift" . 2>/dev/null | grep -v "#if DEBUG\|// DEBUG")
if [ -n "$PRINTS" ]; then
    echo "$PRINTS"
    VIOLATIONS=$((VIOLATIONS + $(echo "$PRINTS" | wc -l)))
else
    echo "  ✅ None found"
fi
echo ""

# Potential hardcoded secrets
echo "🟡 Potential hardcoded secrets:"
SECRETS=$(grep -rEn '"sk-|"pk-|"api_|"secret|"token"' --include="*.swift" . 2>/dev/null | grep -v "enum\|case\|//")
if [ -n "$SECRETS" ]; then
    echo "$SECRETS"
    echo "  ⚠️ Review these manually"
else
    echo "  ✅ None found"
fi
echo ""

# Silent guard returns
echo "🟡 Silent guard returns (may need logging):"
GUARDS=$(grep -rn "else { return }\|else { return nil }" --include="*.swift" . 2>/dev/null | head -10)
if [ -n "$GUARDS" ]; then
    echo "$GUARDS"
    echo "  ⚠️ Review if these need logging"
else
    echo "  ✅ None found"
fi
echo ""

# Logger without privacy
echo "🟡 Logger calls without explicit privacy:"
NO_PRIVACY=$(grep -rn 'Logger\.\w\+\.\w\+("' --include="*.swift" . 2>/dev/null | grep -v "privacy:")
if [ -n "$NO_PRIVACY" ]; then
    echo "$NO_PRIVACY" | head -10
    echo "  ⚠️ Consider adding privacy levels"
else
    echo "  ✅ All have privacy levels"
fi
echo ""

# Singletons
echo "🟡 Singleton pattern (review if needed):"
SINGLETONS=$(grep -rn "static let shared" --include="*.swift" . 2>/dev/null)
if [ -n "$SINGLETONS" ]; then
    echo "$SINGLETONS"
    echo "  ⚠️ Consider Environment injection instead"
else
    echo "  ✅ None found"
fi
echo ""

echo "=== Summary ==="
echo "Critical violations: $VIOLATIONS"
if [ $VIOLATIONS -gt 0 ]; then
    echo "❌ Fix violations before proceeding"
    exit 1
else
    echo "✅ No critical violations"
    exit 0
fi
