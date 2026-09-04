# SwiftUI Architecture Quick Reference

## Cheat Sheet for Common Patterns

---

## ViewModel Pattern (The Three Rules)

### Rule 1: Initialize Once
```swift
// VisionCapture/Sources/App/VisionCaptureApp.swift
@main
struct VisionCaptureApp: App {
    @State private var workflowsVM = WorkflowsViewModel(dataSource: .shared)

    var body: some Scene {
        WindowGroup {
            AppShellView()
                .environment(workflowsVM)
        }
    }
}
```

### Rule 2: Access via Environment
```swift
struct WorkflowsView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel
}
```

### Rule 3: @Bindable for Bindings
```swift
struct EditView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel

    var body: some View {
        @Bindable var vm = viewModel
        TextField("Title", text: $vm.title)
    }
}
```

---

## Property Wrapper Decision Tree

```
┌─────────────────────────────────────────────────────┐
│            Do you need to store a value?            │
└───────────────────────┬─────────────────────────────┘
                        │
          ┌─────────YES─┴─NO──────┐
          ▼                       ▼
┌─────────────────────┐   ┌─────────────────────┐
│  Simple value?      │   │  Receiving from     │
│  (Bool, String)     │   │  parent?            │
└──────────┬──────────┘   └──────────┬──────────┘
           │                         │
    ┌──YES─┴─NO──┐            ┌──YES─┴─NO──┐
    ▼            ▼            ▼            ▼
┌────────┐  ┌────────┐   ┌────────┐   No wrapper
│ @State │  │@Observable│  │Need to │   needed
└────────┘  │ class?  │  │modify? │
            └────┬────┘  └────┬────┘
                 │            │
           ┌─YES─┴─NO─┐  ┌YES─┴─NO─┐
           ▼          ▼  ▼         ▼
      ┌────────┐  Use ┌────────┐ ┌────┐
      │Shared? │struct│@Binding│ │let │
      └────┬───┘      └────────┘ └────┘
           │
     ┌─YES─┴─NO──┐
     ▼           ▼
┌────────────┐ ┌──────────┐
│@Environment│ │Need      │
└────────────┘ │bindings? │
               └────┬─────┘
                    │
              ┌─YES─┴─NO──┐
              ▼           ▼
         ┌─────────┐  ┌─────┐
         │@Bindable│  │ let │
         └─────────┘  └─────┘
```

---

## Quick Reference Table

| Wrapper | When to Use | Example |
|---------|-------------|---------|
| `@State` | View-local simple values | `@State private var isExpanded = false` |
| `@Binding` | Parent passes value to child | `@Binding var selectedItem: Item?` |
| `@Environment` | Shared ViewModel | `@Environment(VM.self) var vm` |
| `@Bindable` | Create bindings from @Observable | `@Bindable var vm = viewModel` |
| `let` | Read-only @Observable | `let viewModel: MyViewModel` |

---

## Stable View Hierarchy

### ✅ CORRECT
```swift
ZStack {
    ContentView().opacity(state == .ideal ? 1 : 0)
    EmptyView().opacity(state == .empty ? 1 : 0)
    LoadingView().opacity(state == .loading ? 1 : 0)
    ErrorView().opacity(state == .error ? 1 : 0)
}
.animation(.snappy(duration: 0.3), value: state)
```

### ❌ WRONG
```swift
if state == .loading {
    LoadingView()
} else if state == .empty {
    EmptyView()
} else {
    ContentView()
}
```

---

## The Five States

| State | Component | Usage |
|-------|-----------|-------|
| **Ideal** | List/Grid | Normal content display |
| **Empty** | `ContentUnavailableView` | No data yet |
| **Loading** | Skeleton/ProgressView | Fetching data |
| **Error** | `ContentUnavailableView` | Something failed |
| **Success** | Toast overlay | Action completed |

```swift
// Empty State
ContentUnavailableView {
    Label("No Items", systemImage: "tray")
} description: {
    Text("Items will appear here.")
} actions: {
    Button("Add Item") { }
        .buttonStyle(.borderedProminent)
}

// Error State
ContentUnavailableView {
    Label("Error", systemImage: "wifi.slash")
} description: {
    Text(errorMessage)
} actions: {
    Button("Try Again") { }
        .buttonStyle(.borderedProminent)
}
```

---

## Navigation Patterns

### NavigationStack
```swift
NavigationStack(path: $path) {
    List { ... }
        .navigationDestination(for: Destination.self) { dest in
            switch dest {
            case .detail(let item): DetailView(item: item)
            }
        }
}
```

### Sheet
```swift
// Boolean
.sheet(isPresented: $showSheet) { SheetView() }

// Item-based
.sheet(item: $selectedItem) { item in DetailView(item: item) }
```

### Confirmation Dialog
```swift
.confirmationDialog("Title", isPresented: $show) {
    Button("Delete", role: .destructive) { }
    Button("Cancel", role: .cancel) { }
}
```

---

## ViewModel Template

```swift
@Observable
class FeatureViewModel: @unchecked Sendable {
    // Dependencies
    private let dataManager: SwiftDataManager

    // UI State
    var showingAddSheet = false
    var errorAlert: ErrorAlert?

    // Search
    var searchText = "" {
        didSet { Task { await updateFiltered() } }
    }

    // Data
    private(set) var filteredItems: [Item] = []
    private var refreshCounter = 0

    // Init
    init(dataSource: SwiftDataManager) {
        self.dataManager = dataSource
    }

    // Actions
    @MainActor
    func updateFiltered() async { ... }

    func createItem() {
        // ...
        refreshCounter += 1
    }
}
```

---

## Animation Quick Reference

### Spring Presets
```swift
.animation(.smooth)                    // Gentle, no bounce
.animation(.snappy(duration: 0.3))     // Quick, responsive ⭐️
.animation(.bouncy(extraBounce: 0.15)) // Playful
```

### Transitions
```swift
.transition(.opacity)
.transition(.move(edge: .bottom))
.transition(.scale)
.transition(.asymmetric(
    insertion: .move(edge: .top).combined(with: .opacity),
    removal: .opacity
))
```

### Symbol Effects (iOS 17+)
```swift
.symbolEffect(.bounce, value: trigger)
.symbolEffect(.pulse, options: .repeating)
.symbolEffect(.wiggle)
.symbolEffect(.rotate)
```

---

## Common Patterns

### Searchable with ViewModel
```swift
struct SearchableView: View {
    @Environment(VM.self) private var viewModel

    var body: some View {
        @Bindable var vm = viewModel

        NavigationStack {
            List { ... }
                .searchable(text: $vm.searchText)
                .onChange(of: viewModel.searchText) { _, query in
                    Task { await viewModel.search(query) }
                }
        }
    }
}
```

### Swipe Actions
```swift
.swipeActions(edge: .trailing) {
    Button(role: .destructive) {
        viewModel.delete(item)
    } label: {
        Label("Delete", systemImage: "trash")
    }
}
.swipeActions(edge: .leading) {
    Button {
        viewModel.toggleFavorite(item)
    } label: {
        Label("Favorite", systemImage: "heart")
    }
    .tint(.pink)
}
```

### Context Menu
```swift
.contextMenu {
    Button { } label: {
        Label("Edit", systemImage: "pencil")
    }
    Button(role: .destructive) { } label: {
        Label("Delete", systemImage: "trash")
    }
}
```

### Toolbar
```swift
.toolbar {
    ToolbarItem(placement: .primaryAction) {
        Button("Add") { }
    }
    ToolbarItem(placement: .topBarLeading) {
        Menu { ... } label: {
            Image(systemName: "ellipsis.circle")
        }
    }
}
```

---

## Forbidden Patterns ❌

```swift
// ❌ Creating ViewModel in view
@State private var viewModel = WorkflowsViewModel()

// ❌ Passing ViewModel as parameter
struct ChildView: View {
    let viewModel: WorkflowsViewModel
}

// ❌ Conditional view swapping
if isLoading { LoadingView() } else { ContentView() }

// ❌ @State with @Observable
@State var viewModel: MyObservableClass

// ❌ Animation without value
.animation(.snappy)  // Should have value:
```

---

## Correct Patterns ✅

```swift
// ✅ Access via Environment
@Environment(WorkflowsViewModel.self) private var viewModel

// ✅ Child also uses Environment
struct ChildView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel
}

// ✅ ZStack with opacity
ZStack {
    LoadingView().opacity(isLoading ? 1 : 0)
    ContentView().opacity(isLoading ? 0 : 1)
}

// ✅ @Bindable for bindings
@Bindable var vm = viewModel
TextField("", text: $vm.searchText)

// ✅ Animation with value
.animation(.snappy(duration: 0.3), value: isLoading)
```

---

## Cross-References

| Topic | Skill |
|-------|-------|
| Async/await, @MainActor | `swift-concurrency-expert` |
| @Observable details | `swift-concurrency-expert` |
| Task management | `swift-concurrency-expert` |
| @Relationship patterns | `swiftdata-relationships` |
| Delete rules | `swiftdata-relationships` |
| Model ownership | `swiftdata-relationships` |

---

## Official Documentation

- [SwiftUI](https://developer.apple.com/documentation/swiftui)
- [Swift Getting Started](https://www.swift.org/getting-started/swiftui/)
- [Observation](https://developer.apple.com/documentation/observation)
- [SwiftData](https://developer.apple.com/documentation/swiftdata)
