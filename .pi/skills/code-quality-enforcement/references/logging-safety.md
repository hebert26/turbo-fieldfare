# Logging Safety

## Anti-Reverse Engineering

Logs are attack vectors. Treat them as public. Protect business logic.

## Use os.Logger

```swift
import os

extension Logger {
    private static var subsystem = Bundle.main.bundleIdentifier ?? "app"
    static let app = Logger(subsystem: subsystem, category: "general")
    static let network = Logger(subsystem: subsystem, category: "network")
    static let data = Logger(subsystem: subsystem, category: "data")
}
```

## Privacy Levels

Always specify privacy on interpolations:

```swift
// Public: Safe to expose
Logger.app.info("Action completed: \(actionName, privacy: .public)")

// Private: Redacted in release logs
Logger.app.debug("User ID: \(userId, privacy: .private)")

// Private with hash: Correlate without exposing
Logger.app.info("Session: \(sessionId, privacy: .private(mask: .hash))")
```

## Never Log

**Secrets:**
- Tokens, passwords, API keys
- Credit card numbers, SSN
- Encryption keys

**Business logic:**
- Fraud scores, thresholds, factors
- Pricing algorithms, multipliers
- Validation rules, decision trees

```swift
// ❌ Violation: Exposes business logic
Logger.app.debug("Fraud score: \(score), threshold: \(threshold)")
Logger.app.info("Price: base=\(base), mult=\(mult), disc=\(discount)")

// ✅ Fix: Opaque logging
Logger.app.debug("Validation: \(isValid ? "pass" : "fail", privacy: .public)")
Logger.app.info("Calculation completed")
```

## Replace print()

```swift
// ❌ Violation
print("User token: \(token)")
print("Processing \(item)")

// ✅ Fix
Logger.app.debug("Processing item: \(item.id, privacy: .private)")
// Never log tokens at all
```

## Optional: Strip Logs in Release

For maximum security:

```swift
#if DEBUG
import os
let log = Logger(subsystem: "app", category: "debug")
#else
struct NoOpLogger {
    func debug(_ message: String) {}
    func info(_ message: String) {}
    func error(_ message: String) {}
}
let log = NoOpLogger()
#endif
```

## Validation Commands

```bash
# Find print statements
grep -rn "print(" --include="*.swift" .

# Find potential secret logging
grep -rEn "Logger.*token|Logger.*password|Logger.*key|Logger.*secret" --include="*.swift" .

# Find logging without privacy
grep -rn "Logger\." --include="*.swift" . | grep -v "privacy:"
```
