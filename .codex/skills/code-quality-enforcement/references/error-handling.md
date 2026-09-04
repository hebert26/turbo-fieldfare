# Error Handling

## Mandatory Rules

Every method that can fail MUST handle errors meaningfully. No exceptions.

## Violations to Fix

**Empty catch:**
```swift
// ❌ Violation
do { try operation() } catch { }

// ✅ Fix: Propagate
func perform() throws {
    do { try operation() } 
    catch { throw AppError.operationFailed(underlying: error) }
}

// ✅ Fix: User feedback
do { try operation() } 
catch { 
    errorMessage = "Unable to complete. Please try again."
    showError = true 
}
```

**Silent guard return:**
```swift
// ❌ Violation
guard let data = optionalData else { return }

// ✅ Fix: Log and/or throw
guard let data = optionalData else {
    Logger.app.warning("Missing required data")
    throw AppError.missingData
}
```

**Catch with only print:**
```swift
// ❌ Violation
catch { print("Error: \(error)") }

// ✅ Fix: Use Logger + handle
catch {
    Logger.app.error("Operation failed: \(error.localizedDescription, privacy: .public)")
    throw AppError.operationFailed(underlying: error)
}
```

## Required Patterns

**Typed error enum:**
```swift
enum AppError: LocalizedError {
    case networkUnavailable
    case dataCorrupted(details: String)
    case operationFailed(underlying: Error)
    
    var errorDescription: String? {
        switch self {
        case .networkUnavailable: return "Network unavailable"
        case .dataCorrupted(let d): return "Data error: \(d)"
        case .operationFailed: return "Operation failed"
        }
    }
}
```

**Result for callbacks:**
```swift
func fetch(completion: @escaping (Result<Data, AppError>) -> Void)
```

## Validation Commands

```bash
# Find empty catches
grep -rn "catch { }" --include="*.swift" .
grep -rn "catch {$" --include="*.swift" .

# Find guard with just return
grep -rn "else { return }" --include="*.swift" .
```
