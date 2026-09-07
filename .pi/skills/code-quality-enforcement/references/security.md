# Security

## Hardcoded Secrets

**Never hardcode:** API keys, tokens, passwords, secrets.

```swift
// ❌ Violation
let apiKey = "sk-1234567890"

// ✅ Fix: Keychain
let apiKey = try KeychainManager.retrieve(.apiKey)
```

## Keychain

Use `kSecAttrAccessibleWhenUnlockedThisDeviceOnly` for maximum security.

```swift
final class KeychainManager {
    enum Key: String {
        case authToken, apiKey, encryptionKey
    }
    
    static func store(_ data: Data, for key: Key) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key.rawValue,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]
        SecItemDelete(query as CFDictionary)
        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw AppError.keychainError(status: status)
        }
    }
    
    static func retrieve(_ key: Key) throws -> Data {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key.rawValue,
            kSecReturnData as String: true
        ]
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess, let data = result as? Data else {
            throw AppError.keychainError(status: status)
        }
        return data
    }
}
```

## Input Validation

Validate all external input before use.

```swift
func validate(_ input: String) throws -> String {
    let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { throw AppError.emptyInput }
    guard trimmed.count <= 1000 else { throw AppError.inputTooLong }
    return trimmed
}
```

## Debug Code

No debug code in release builds.

```swift
#if DEBUG
func debugLog(_ message: String) { print("[DEBUG] \(message)") }
#endif
```

## Validation Commands

```bash
# Find potential hardcoded secrets
grep -rEn '"\w{20,}"' --include="*.swift" .
grep -rEn '"sk-|"pk-|"api_|"secret' --include="*.swift" .
grep -rn 'apiKey\|secret\|password' --include="*.swift" . | grep -v "Keychain\|enum\|case"

# Find print in non-debug context
grep -rn "print(" --include="*.swift" . | grep -v "#if DEBUG"
```
