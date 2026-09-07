#!/bin/bash
# Discover existing patterns in the codebase

echo "=== Pattern Discovery ==="
echo ""

# Find Swift files
echo "📁 Swift files:"
find . -name "*.swift" -type f | head -20
echo ""

# Architecture patterns
echo "🏗️ Architecture patterns:"
echo "  @Observable classes:"
grep -rln "@Observable" --include="*.swift" . 2>/dev/null | wc -l | xargs echo "   "

echo "  Legacy ObservableObject (should be 0):"
grep -rln "ObservableObject" --include="*.swift" . 2>/dev/null | wc -l | xargs echo "   "

echo "  Actors:"
grep -rln "^actor \|^actor " --include="*.swift" . 2>/dev/null | wc -l | xargs echo "   "

echo "  @MainActor classes:"
grep -rln "@MainActor" --include="*.swift" . 2>/dev/null | wc -l | xargs echo "   "
echo ""

# Error handling patterns
echo "🚨 Error handling:"
echo "  Try/catch blocks:"
grep -rn "catch {" --include="*.swift" . 2>/dev/null | wc -l | xargs echo "   "

echo "  Result types:"
grep -rn "Result<" --include="*.swift" . 2>/dev/null | wc -l | xargs echo "   "

echo "  Custom Error enums:"
grep -rn ": Error\|: LocalizedError" --include="*.swift" . 2>/dev/null | wc -l | xargs echo "   "
echo ""

# Logging patterns
echo "📝 Logging:"
echo "  os.Logger usage:"
grep -rn "Logger\." --include="*.swift" . 2>/dev/null | wc -l | xargs echo "   "

echo "  print() statements:"
grep -rn "print(" --include="*.swift" . 2>/dev/null | wc -l | xargs echo "   "
echo ""

# Security patterns
echo "🔐 Security:"
echo "  Keychain usage:"
grep -rn "kSecClass\|SecItem" --include="*.swift" . 2>/dev/null | wc -l | xargs echo "   "

echo "  Potential hardcoded strings (20+ chars):"
grep -rEn '"\w{20,}"' --include="*.swift" . 2>/dev/null | wc -l | xargs echo "   "
echo ""

echo "=== Discovery Complete ==="
