#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
APP_NAME="TurboFieldfare"
APP_PROCESS="TurboFieldfareMac"
SERVICE_PROCESS="TurboFieldfareDecodeService"
BUNDLE_ID="com.turbofieldfare.TurboFieldfare"
MIN_SYSTEM_VERSION="26.0"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
STAGE_BUNDLE="$DIST_DIR/$APP_NAME.app"
APP_CONTENTS="$STAGE_BUNDLE/Contents"
APP_MACOS="$APP_CONTENTS/MacOS"
APP_RESOURCES="$APP_CONTENTS/Resources"
INFO_PLIST="$APP_CONTENTS/Info.plist"
INSTALL_BUNDLE="/Applications/$APP_NAME.app"
INSTALLED_RESOURCES="$INSTALL_BUNDLE/Contents/Resources"

case "$MODE" in
  run|--verify|verify|--stage-only|stage-only)
    ;;
  *)
    echo "usage: $0 [run|--verify|--stage-only]" >&2
    exit 2
    ;;
esac

stop_executable() {
  /usr/bin/pkill -f "(^|/)$1( |$)" >/dev/null 2>&1 || true
}

swift build -c release --product "$APP_PROCESS"
swift build -c release --product "$SERVICE_PROCESS"
BIN_DIR="$(swift build -c release --show-bin-path)"
APP_VERSION="$(/usr/bin/sed -nE 's/.*fallbackShortVersion = "([^"]+)".*/\1/p' \
  "$ROOT_DIR/Sources/TurboFieldfareApp/MacPresentation/AboutPanelPresentation.swift" \
  | /usr/bin/head -n 1)"
if [[ -z "$APP_VERSION" ]]; then
  echo "could not determine the TurboFieldfare app version" >&2
  exit 1
fi

/bin/rm -rf "$STAGE_BUNDLE"
/bin/mkdir -p "$APP_MACOS" "$APP_RESOURCES"
/usr/bin/ditto "$BIN_DIR/$APP_PROCESS" "$APP_MACOS/$APP_PROCESS"
/usr/bin/ditto "$BIN_DIR/$SERVICE_PROCESS" "$APP_MACOS/$SERVICE_PROCESS"
/bin/chmod +x "$APP_MACOS/$APP_PROCESS" "$APP_MACOS/$SERVICE_PROCESS"

for resource_bundle in "$BIN_DIR"/*.bundle; do
  bundle_name="$(/usr/bin/basename "$resource_bundle")"
  /usr/bin/ditto "$resource_bundle" "$APP_RESOURCES/$bundle_name"
done

rewrite_swiftpm_resource_paths() {
  /usr/bin/ruby -e '
    binary, build_dir, staged_resources, installed_resources = ARGV
    bytes = File.binread(binary)
    pattern = Regexp.new(
      Regexp.escape(build_dir.b) + %q{/[A-Za-z0-9_-]+[.]bundle})
    sources = bytes.scan(pattern).uniq
    abort "no SwiftPM resource paths found in #{binary}" if sources.empty?

    sources.each do |source|
      bundle_name = File.basename(source)
      staged_bundle = File.join(staged_resources, bundle_name)
      abort "missing staged resource bundle #{bundle_name}" unless File.directory?(staged_bundle)

      installed_path = File.join(installed_resources, bundle_name)
      if installed_path.bytesize > source.bytesize
        abort "installed resource path is longer than the compiled SwiftPM path"
      end
      replacement = installed_path + ("/" * (source.bytesize - installed_path.bytesize))
      bytes.gsub!(source, replacement)
    end

    File.binwrite(binary, bytes)
  ' "$1" "$BIN_DIR" "$APP_RESOURCES" "$INSTALLED_RESOURCES"
}

rewrite_swiftpm_resource_paths "$APP_MACOS/$APP_PROCESS"
rewrite_swiftpm_resource_paths "$APP_MACOS/$SERVICE_PROCESS"

/usr/bin/plutil -create xml1 "$INFO_PLIST"
/usr/bin/plutil -insert CFBundleDisplayName -string "$APP_NAME" "$INFO_PLIST"
/usr/bin/plutil -insert CFBundleExecutable -string "$APP_PROCESS" "$INFO_PLIST"
/usr/bin/plutil -insert CFBundleIdentifier -string "$BUNDLE_ID" "$INFO_PLIST"
/usr/bin/plutil -insert CFBundleName -string "$APP_NAME" "$INFO_PLIST"
/usr/bin/plutil -insert CFBundlePackageType -string APPL "$INFO_PLIST"
/usr/bin/plutil -insert CFBundleShortVersionString -string "$APP_VERSION" "$INFO_PLIST"
/usr/bin/plutil -insert CFBundleVersion -string "$APP_VERSION" "$INFO_PLIST"
/usr/bin/plutil -insert LSMinimumSystemVersion -string "$MIN_SYSTEM_VERSION" "$INFO_PLIST"
/usr/bin/plutil -insert NSHighResolutionCapable -bool true "$INFO_PLIST"
/usr/bin/plutil -insert NSPrincipalClass -string NSApplication "$INFO_PLIST"
/usr/bin/codesign --force --deep --sign - "$STAGE_BUNDLE"
/usr/bin/codesign --verify --deep --strict "$STAGE_BUNDLE"

if [[ "$MODE" == "--stage-only" || "$MODE" == "stage-only" ]]; then
  echo "staged and verified $STAGE_BUNDLE"
  exit 0
fi

stop_executable "$APP_PROCESS"
# Older hand-built bundles used the display name as the executable name.
/usr/bin/pkill -f "^$INSTALL_BUNDLE/Contents/MacOS/$APP_NAME( |$)" \
  >/dev/null 2>&1 || true
stop_executable "$SERVICE_PROCESS"

install_parent="$(/usr/bin/dirname "$INSTALL_BUNDLE")"
install_name="$(/usr/bin/basename "$INSTALL_BUNDLE")"
install_candidate="$install_parent/.$install_name.installing.$$"
install_previous="$install_parent/.$install_name.previous.$$"
/bin/rm -rf "$install_candidate" "$install_previous"
/usr/bin/ditto "$STAGE_BUNDLE" "$install_candidate"
if [[ -e "$INSTALL_BUNDLE" ]]; then
  /bin/mv "$INSTALL_BUNDLE" "$install_previous"
fi
if ! /bin/mv "$install_candidate" "$INSTALL_BUNDLE"; then
  if [[ -e "$install_previous" ]]; then
    /bin/mv "$install_previous" "$INSTALL_BUNDLE"
  fi
  exit 1
fi

verify_installed_bundle() {
  /usr/bin/codesign --verify --deep --strict "$INSTALL_BUNDLE" || return 1
  local installed_executable
  installed_executable="$(/usr/libexec/PlistBuddy \
    -c 'Print :CFBundleExecutable' "$INSTALL_BUNDLE/Contents/Info.plist")" \
    || return 1
  [[ "$installed_executable" == "$APP_PROCESS" ]] || return 1
  [[ -x "$INSTALL_BUNDLE/Contents/MacOS/$APP_PROCESS" ]] || return 1
  [[ -x "$INSTALL_BUNDLE/Contents/MacOS/$SERVICE_PROCESS" ]] || return 1
  /usr/bin/cmp -s "$STAGE_BUNDLE/Contents/MacOS/$APP_PROCESS" \
    "$INSTALL_BUNDLE/Contents/MacOS/$APP_PROCESS" || return 1
  /usr/bin/cmp -s "$STAGE_BUNDLE/Contents/MacOS/$SERVICE_PROCESS" \
    "$INSTALL_BUNDLE/Contents/MacOS/$SERVICE_PROCESS" || return 1
}

if ! verify_installed_bundle; then
  echo "the installed TurboFieldfare bundle is incomplete" >&2
  /bin/rm -rf "$INSTALL_BUNDLE"
  if [[ -e "$install_previous" ]]; then
    /bin/mv "$install_previous" "$INSTALL_BUNDLE"
  fi
  exit 1
fi
/bin/rm -rf "$install_previous"

/usr/bin/open -n "$INSTALL_BUNDLE"

if [[ "$MODE" == "--verify" || "$MODE" == "verify" ]]; then
  for _ in {1..20}; do
    if /usr/bin/pgrep -f "^$INSTALL_BUNDLE/Contents/MacOS/$APP_PROCESS( |$)" \
        >/dev/null; then
      exit 0
    fi
    /bin/sleep 0.25
  done
  echo "$APP_PROCESS did not start" >&2
  exit 1
fi
