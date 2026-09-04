# ViewModel Pattern Reference

## VisionCapture's Environment-Based ViewModel Architecture

This document provides comprehensive guidance on VisionCapture's ViewModel pattern, which differs significantly from traditional MVVM.

---

## Overview: Why Not Traditional MVVM?

Traditional MVVM creates ViewModels in views:
```swift
// ❌ Traditional MVVM - DON'T USE
struct WorkflowsView: View {
    @StateObject var viewModel = WorkflowsViewModel()
}
```

**Problems with this approach**:
1. **Multiple Instances**: Each navigation creates a new ViewModel
2. **State Loss**: ViewModel state resets on navigation
3. **No Sharing**: Can't share state between sibling views
4. **Inefficient**: Duplicate data fetching and processing

**VisionCapture's Solution**: Environment-based ViewModels as app-level singletons.

---

## The Three Rules Explained

### Rule 1: Initialize Once at App Startup

**Where**: `VisionCaptureApp.swift`

**How**: Use `@State` in the App struct to hold ViewModels.

```swift
@main
struct VisionCaptureApp: App {
    // MARK: - Data Layer
    private let dataManager: SwiftDataManager

    // MARK: - ViewModels (initialized once)
    @State private var workflowsViewModel: WorkflowsViewModel
    @State private var captureViewModel: CaptureViewModel
    @State private var explorationViewModel: ExplorationViewModel

    init() {
        // Initialize data manager
        let manager = SwiftDataManager.shared
        self.dataManager = manager

        // Initialize ViewModels with dependencies
        _workflowsViewModel = State(initialValue: WorkflowsViewModel(dataSource: manager))
        _captureViewModel = State(initialValue: CaptureViewModel(dataSource: manager))
        _explorationViewModel = State(initialValue: ExplorationViewModel(dataSource: manager))
    }

    var body: some Scene {
        WindowGroup {
            AppShellView()
                // Inject into environment
                .environment(workflowsViewModel)
                .environment(captureViewModel)
                .environment(explorationViewModel)
                .environment(dataManager)
        }
    }
}
```

**Why @State in App**:
- App struct is the root, created once
- @State ensures SwiftUI manages lifecycle
- ViewModels persist for entire app session

---

### Rule 2: Access via @Environment

**Where**: Any view that needs the ViewModel

**How**: Use `@Environment(Type.self)` to retrieve.

```swift
struct WorkflowsView: View {
    // Access the shared ViewModel
    @Environment(WorkflowsViewModel.self) private var viewModel

    var body: some View {
        NavigationStack {
            List(viewModel.filteredWorkflows) { workflow in
                WorkflowRow(workflow: workflow)
            }
            .navigationTitle("Workflows")
            .task {
                await viewModel.updateFilteredWorkflows()
            }
        }
    }
}
```

**Benefits**:
- Any view in hierarchy can access
- No prop drilling through intermediary views
- Single source of truth maintained
- SwiftUI handles observation automatically

**Child Views Also Use @Environment**:
```swift
struct WorkflowRow: View {
    let workflow: Workflow

    // Child can also access ViewModel directly
    @Environment(WorkflowsViewModel.self) private var viewModel

    var body: some View {
        HStack {
            Text(workflow.title)
            Spacer()
            Button {
                viewModel.toggleFavorite(workflow)
            } label: {
                Image(systemName: workflow.isFavorite ? "heart.fill" : "heart")
            }
        }
    }
}
```

---

### Rule 3: @Bindable for Bindings

**When**: View needs two-way bindings to ViewModel properties (TextField, Toggle, etc.)

**How**: Create `@Bindable` wrapper in view body.

```swift
struct SearchableWorkflowsView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel

    var body: some View {
        // Create bindable wrapper for bindings
        @Bindable var bindableVM = viewModel

        NavigationStack {
            WorkflowsList()
                .searchable(text: $bindableVM.searchText)  // Two-way binding
        }
    }
}
```

**Alternative: Child with @Bindable parameter**:
```swift
struct EditWorkflowView: View {
    @Bindable var viewModel: WorkflowsViewModel

    var body: some View {
        Form {
            TextField("Title", text: $viewModel.workflowTitle)
            TextField("Intent", text: $viewModel.workflowIntent)
            TextField("Notes", text: $viewModel.workflowNotes)
        }
    }
}

// Parent passes viewModel
struct ParentView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel

    var body: some View {
        EditWorkflowView(viewModel: viewModel)
    }
}
```

---

## ViewModel Structure Template

```swift
import SwiftUI
import SwiftData
import OSLog

/// ViewModel for managing [Feature] business logic
/// Access via: @Environment(FeatureViewModel.self)
@Observable
class FeatureViewModel: @unchecked Sendable {

    // MARK: - Dependencies
    private let logger = Logger(subsystem: "com.app", category: "Feature")
    private let dataManager: SwiftDataManager

    // MARK: - UI State (drives view updates)
    var showingAddSheet = false
    var showingEditSheet = false
    var errorAlert: ErrorAlert?
    var successMessage: String?
    var showingSuccessToast = false

    // MARK: - Selection State
    var selectedItem: Item?
    var selectedItems: Set<Item> = []

    // MARK: - Search State
    var searchText: String = "" {
        didSet {
            isSearching = !searchText.isEmpty
            Task { await updateFiltered() }
        }
    }
    var isSearching = false

    // MARK: - Filter State
    var selectedFilter: FilterOption = .all {
        didSet { Task { await updateFiltered() } }
    }
    var sortOrder: SortOrder = .dateDescending {
        didSet { Task { await updateFiltered() } }
    }

    // MARK: - Computed Data (async updated)
    private(set) var filteredItems: [Item] = []

    // MARK: - Refresh Counter
    /// Forces SwiftUI to re-evaluate computed properties when incremented
    private var refreshCounter = 0

    // MARK: - Background Tasks
    private var filterTask: Task<Void, Never>?
    private var searchTask: Task<Void, Never>?

    // MARK: - Computed Properties
    var hasActiveFilters: Bool {
        _ = refreshCounter  // Track dependency
        return selectedFilter != .all || !searchText.isEmpty
    }

    var itemCount: Int {
        _ = refreshCounter
        return filteredItems.count
    }

    // MARK: - Initialization
    init(dataSource: SwiftDataManager) {
        self.dataManager = dataSource
        Task { await updateFiltered() }
    }

    // MARK: - Data Loading
    @MainActor
    func updateFiltered() async {
        filterTask?.cancel()

        filterTask = Task {
            // Debounce rapid changes
            try? await Task.sleep(for: .milliseconds(50))
            guard !Task.isCancelled else { return }

            // Perform filtering (can be off main thread for heavy work)
            let items = await loadAndFilterItems()

            guard !Task.isCancelled else { return }

            // Update on main thread
            filteredItems = items
        }
    }

    private func loadAndFilterItems() async -> [Item] {
        // Load and filter logic here
        return []
    }

    // MARK: - Actions
    func createItem(title: String, notes: String) {
        do {
            try dataManager.createItem(title: title, notes: notes)
            showSuccess("Item created")
            refreshCounter += 1
        } catch {
            showError(error)
        }
    }

    func deleteItem(_ item: Item) {
        do {
            try dataManager.deleteItem(item)
            showSuccess("Item deleted")
            refreshCounter += 1
        } catch {
            showError(error)
        }
    }

    func toggleFavorite(_ item: Item) {
        do {
            try dataManager.toggleFavorite(item)
            refreshCounter += 1
        } catch {
            showError(error)
        }
    }

    // MARK: - UI Helpers
    private func showSuccess(_ message: String) {
        successMessage = message
        showingSuccessToast = true
        UINotificationFeedbackGenerator().notificationOccurred(.success)

        // Auto-hide
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            withAnimation {
                self.showingSuccessToast = false
            }
        }
    }

    private func showError(_ error: Error) {
        errorAlert = ErrorAlert(
            title: "Error",
            message: error.localizedDescription
        )
        UINotificationFeedbackGenerator().notificationOccurred(.error)
    }

    // MARK: - Reset
    func resetFilters() {
        searchText = ""
        selectedFilter = .all
        sortOrder = .dateDescending
    }
}

// MARK: - Supporting Types
struct ErrorAlert: Identifiable {
    let id = UUID()
    let title: String
    let message: String
}

enum FilterOption: String, CaseIterable, Identifiable {
    case all = "All"
    case favorites = "Favorites"
    case recent = "Recent"

    var id: String { rawValue }
}

enum SortOrder {
    case dateAscending
    case dateDescending
    case titleAscending
    case titleDescending
}
```

---

## Key Patterns Explained

### 1. @Observable vs ObservableObject

```swift
// ✅ Modern: @Observable (iOS 17+)
@Observable
class ViewModel {
    var items: [Item] = []  // No @Published needed
}

// ❌ Legacy: ObservableObject
class ViewModel: ObservableObject {
    @Published var items: [Item] = []  // Required @Published
}
```

**Why @Observable**:
- No Combine dependency
- Granular observation (only accessed properties tracked)
- More efficient view updates
- Simpler syntax

### 2. @unchecked Sendable

```swift
@Observable
class ViewModel: @unchecked Sendable {
    // ...
}
```

**Why**:
- ViewModels are shared across threads
- `@unchecked Sendable` tells compiler we handle thread safety
- Combined with `@MainActor` on async methods for UI updates

### 3. refreshCounter Pattern

```swift
private var refreshCounter = 0

var computedProperty: [Item] {
    _ = refreshCounter  // Track dependency
    return items.filter { ... }
}

func modifyData() {
    // ... modify data ...
    refreshCounter += 1  // Force re-computation
}
```

**Why**:
- SwiftUI doesn't auto-track computed property dependencies
- Incrementing counter forces SwiftUI to re-evaluate
- Alternative to maintaining separate `@Published` filtered arrays

### 4. Async Filter Pattern

```swift
var searchText: String = "" {
    didSet {
        Task { await updateFiltered() }
    }
}

@MainActor
func updateFiltered() async {
    filterTask?.cancel()

    filterTask = Task {
        try? await Task.sleep(for: .milliseconds(50))  // Debounce
        guard !Task.isCancelled else { return }

        // Heavy work
        let filtered = await performFiltering()

        guard !Task.isCancelled else { return }
        filteredItems = filtered
    }
}
```

**Benefits**:
- Non-blocking UI
- Automatic debouncing
- Cancels outdated requests
- Clean main thread

---

## Migration from Traditional MVVM

### Before (Traditional)
```swift
// View creates its own ViewModel
struct WorkflowsView: View {
    @StateObject var viewModel = WorkflowsViewModel()

    var body: some View {
        // Child needs ViewModel passed explicitly
        WorkflowDetail(viewModel: viewModel, workflow: selected)
    }
}

struct WorkflowDetail: View {
    @ObservedObject var viewModel: WorkflowsViewModel
    let workflow: Workflow
}
```

### After (VisionCapture Pattern)
```swift
// App creates ViewModel
@main
struct App: App {
    @State private var workflowsVM = WorkflowsViewModel(...)

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(workflowsVM)
        }
    }
}

// Views access via Environment
struct WorkflowsView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel

    var body: some View {
        // Child accesses via Environment too
        WorkflowDetail(workflow: selected)
    }
}

struct WorkflowDetail: View {
    @Environment(WorkflowsViewModel.self) private var viewModel
    let workflow: Workflow
}
```

---

## Real VisionCapture Examples

### WorkflowsViewModel (Abbreviated)
**File**: `VisionCapture/Sources/Features/Workflows/ViewModel/WorkflowsViewModel.swift`

```swift
@Observable class WorkflowsViewModel: @unchecked Sendable {
    private let dataManager: SwiftDataManager

    // UI State
    var showingAddWorkflow = false
    var errorAlert: ErrorAlert?
    var workflowToDelete: Workflow?

    // Search
    var searchText: String = "" {
        didSet {
            isSearching = !searchText.isEmpty
        }
    }
    var isSearching = false
    var semanticSearchResults: [CoreSearchResult] = []

    // Filters
    var isFavoritesActive = false
    var isPinnedActive = false
    var selectedSortOption: WorkflowSortOption = .dateAdded

    // Computed (async updated)
    private(set) var filteredWorkflows: [Workflow] = []

    // Refresh
    private var refreshCounter = 0

    init(dataSource: SwiftDataManager, ...) {
        self.dataManager = dataSource
        Task { await updateFilteredWorkflows() }
    }

    @MainActor
    func updateFilteredWorkflows() async { ... }

    @MainActor
    func performSemanticSearch(_ query: String) async { ... }

    func toggleFavorite(_ workflow: Workflow) { ... }
    func togglePin(_ workflow: Workflow) { ... }
    func deleteWorkflow(_ workflow: Workflow) { ... }
}
```

### Usage in View
**File**: `VisionCapture/Sources/Features/Workflows/Views/WorkflowsHubView.swift`

```swift
struct WorkflowsHubView: View {
    @Environment(WorkflowsViewModel.self) private var viewModel

    var body: some View {
        @Bindable var bindableVM = viewModel

        NavigationStack {
            WorkflowsListView()
                .searchable(text: $bindableVM.searchText)
                .onChange(of: viewModel.searchText) { _, newValue in
                    Task {
                        await viewModel.performSemanticSearch(newValue)
                    }
                }
        }
    }
}
```

---

## Common Pitfalls

### 1. Forgetting to Inject in App
```swift
// ❌ Forgot .environment()
var body: some Scene {
    WindowGroup {
        ContentView()
        // Missing: .environment(viewModel)
    }
}

// Result: @Environment fails at runtime
```

### 2. Using @State with @Observable
```swift
// ❌ Wrong wrapper
struct MyView: View {
    @State var viewModel: WorkflowsViewModel  // NO!
}

// ✅ Correct
struct MyView: View {
    @Environment(WorkflowsViewModel.self) var viewModel
}
```

### 3. Creating Bindings Without @Bindable
```swift
// ❌ Can't create binding directly
@Environment(VM.self) var vm

TextField("", text: $vm.text)  // ERROR!

// ✅ Use @Bindable
@Bindable var bindable = vm
TextField("", text: $bindable.text)  // Works
```

---

## Summary

| Aspect | Traditional MVVM | VisionCapture Pattern |
|--------|-----------------|------------------|
| **Creation** | In each view | Once in App |
| **Access** | Parameter passing | @Environment |
| **Bindings** | @ObservedObject | @Bindable |
| **Scope** | Per-view | App-wide |
| **State** | Resets on navigation | Persists |
| **Sharing** | Difficult | Built-in |
