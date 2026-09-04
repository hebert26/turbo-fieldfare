# Apple Patterns (iOS 26)

## @Observable

**Required:** `@Observable` macro on model classes.

```swift
@Observable
class Model {
    var value: String = ""
}
```

**Ownership:** `@State` only where instance is created. Children receive without wrapper or with `@Bindable`.

```swift
struct Parent: View {
    @State private var model = Model()  // Owner
    var body: some View { Child(model: model) }
}

struct Child: View {
    let model: Model          // Read-only
    // OR
    @Bindable var model: Model  // Read-write bindings
}
```

## Environment

Inject via `.environment(model)`, retrieve via `@Environment(Model.self)`.

## Concurrency (Swift 6.2)

**Default @MainActor isolation:** Enabled in Xcode 26 projects. Code runs on MainActor unless marked otherwise.

```swift
// Runs on MainActor by default
func updateUI() { }

// Explicit concurrent execution
@concurrent
func fetchData() async { }

// Inherit caller's actor
nonisolated func helper() { }
```

**Actors:** Use for shared mutable state accessed from multiple contexts.

```swift
actor DataCache {
    private var cache: [String: Data] = [:]
    func store(_ data: Data, for key: String) { cache[key] = data }
}
```

**@MainActor:** Required for ModelContext operations and UI updates.

```swift
@MainActor
final class StorageManager {
    private let context: ModelContext
}
```

## SwiftData

**Versioned schema:**
```swift
enum SchemaV1: VersionedSchema {
    static var versionIdentifier = Schema.Version(1, 0, 0)
    static var models: [any PersistentModel.Type] { [Item.self] }
}
```

**Explicit URL for App Groups:**
```swift
let config = ModelConfiguration(
    schema: Schema(versionedSchema: SchemaV1.self),
    url: containerURL.appendingPathComponent("store.sqlite")
)
```

## Validation Commands

```bash
# Check for legacy patterns
grep -rn "ObservableObject\|@Published\|@StateObject\|@ObservedObject" --include="*.swift" .

# Check for singleton anti-pattern
grep -rn "static let shared" --include="*.swift" .
```
