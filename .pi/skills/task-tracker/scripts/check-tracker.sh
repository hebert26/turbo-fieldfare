#!/usr/bin/env bash
# check-tracker.sh - run every check in references/self-check.md against a work-item folder.
#
#   check-tracker.sh <work-item-folder> [--base <sha>]
#
# Prints PASS / WARN / FAIL per check. Exits non-zero if anything FAILed.
# A non-zero exit means you may not hand over.
#
# Written for bash, so it behaves the same when called from zsh (this machine's
# default shell). Never rewrite the pathspec arrays as plain strings - zsh does
# not word-split them, git then matches nothing, and the D4 gate turns green on
# untested code. See references/coverage-gate.md.

set -u

DIR="${1:-.}"
[ "${DIR#--}" != "$DIR" ] && DIR="."
BASE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE="${2:-}"; shift 2 ;;
    *)      shift ;;
  esac
done

# Source paths D4 watches. An array, never a string.
PATHS=(VisionCapture/Sources VisionCapture/Resources VisionCapture/scripts VisionCapture/Package.swift)

# Words banned by references/field-values.md. Literal ones only.
BANNED='works correctly|as expected|\bTBD\b|\betc\b|exit criteria|Definition of Done'

fails=0
warns=0
pass() { printf '  PASS  %s\n' "$1"; }
warn() { printf '  WARN  %s\n' "$1"; warns=$((warns+1)); }
fail() { printf '  FAIL  %s\n' "$1"; fails=$((fails+1)); }
head2() { printf '\n%s\n' "$1"; }

cd "$DIR" 2>/dev/null || { echo "no such folder: $DIR"; exit 2; }
REPO=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "not a git repo"; exit 2; }

T=$(ls tracker-*.md            2>/dev/null | head -1)
I=$(ls implementation-*.md     2>/dev/null | head -1)

# The owner's page lives OUTSIDE the work-item folder, in project-files/human/,
# so an agent listing this folder never sees it. ADR 0052. Its path is derived
# from the slug - nothing in either Markdown file points at it.
SLUG=$(sed -n 's/^slug:[[:space:]]*//p' "$T" 2>/dev/null | tr -d '"' | head -1)
[ -n "$SLUG" ] || SLUG=$(basename "$PWD")
PF="$PWD"; while [ "$PF" != "/" ] && [ "$(basename "$PF")" != "project-files" ]; do PF=$(dirname "$PF"); done
H="$PF/human/$SLUG.html"
STRAY=$(ls implementation-*.html 2>/dev/null | head -1)

echo "check-tracker  $(pwd)"
[ -n "$T" ] || { echo "  FAIL  no tracker-*.md here"; exit 1; }

PHASES=$(grep -c '^## Phase '  "$T")
TASKS=$(grep -cE '^- \[.\] [0-9]+\.[0-9]+ ' "$T")
LINES=$(wc -l < "$T" | tr -d ' ')

# Split the tracker into one file per phase so per-phase checks are honest.
CHUNKS=$(mktemp -d)
trap 'rm -rf "$CHUNKS"' EXIT
awk -v d="$CHUNKS" '
  /^## Phase / { n++ }
  n > 0 { print > (d "/p" n) }
' "$T"

# ---------------------------------------------------------------- 1. size
head2 "1. size"
BUDGET=$(( 60 + 30 * PHASES + TASKS ))
LOW=$(( BUDGET * 85 / 100 )); HIGH=$(( BUDGET * 115 / 100 ))
if [ "$LINES" -ge "$LOW" ] && [ "$LINES" -le "$HIGH" ]; then
  pass "$LINES lines, budget $BUDGET ($PHASES phases, $TASKS tasks)"
elif [ "$LINES" -gt "$HIGH" ]; then
  warn "$LINES lines vs budget $BUDGET - prose has leaked in, move it to $I"
else
  warn "$LINES lines vs budget $BUDGET - too short, a phase is probably missing a block"
fi

# ------------------------------------------------- 2. no per-task subsections
head2 "2. no per-task subsections"
if grep -q '^####' "$T"; then
  fail "$(grep -c '^####' "$T") '####' headings - one task is one line"
else
  pass "none"
fi

# ------------------------------------------------ 3. deleted concepts + banned
head2 "3. deleted concepts and banned words"
hits=$(grep -niE "$BANNED" "$T" | head -5)
if [ -n "$hits" ]; then
  fail "banned wording in the tracker:"; printf '%s\n' "$hits" | sed 's/^/        /'
else
  pass "none"
fi

# ------------------------------------------------------- 4. the gate exists
head2 "4. every phase has all four blocks"
c_cov=$(grep -c '^\*\*Unit test coverage\*\*' "$T")
c_don=$(grep -c '^\*\*Done when\*\*'          "$T")
c_acc=$(grep -c '^\*\*Acceptance\*\*'         "$T")
if [ "$PHASES" -eq "$c_cov" ] && [ "$PHASES" -eq "$c_don" ] && [ "$PHASES" -eq "$c_acc" ]; then
  pass "$PHASES phases, $PHASES of each block"
else
  fail "phases=$PHASES coverage=$c_cov done-when=$c_don acceptance=$c_acc - a block was dropped"
fi

# -------------------------------------------------------- 5. five D lines
head2 "5. five done-when lines per phase"
d=$(grep -c '^- \[.\] D[1-5] ' "$T")
if [ "$d" -eq $(( PHASES * 5 )) ]; then
  pass "$d lines for $PHASES phases"
else
  fail "$d D-lines, expected $(( PHASES * 5 )) - someone added or deleted one"
fi

# ------------------------------------------------------ 6. gate not reworded
head2 "6. nobody reworded D3 or D4"
for n in 3 4; do
  u=$(grep -h "^- \[.\] D$n " "$T" | sed "s/.*D$n //" | sort -u | wc -l | tr -d ' ')
  if [ "$u" -le 1 ]; then pass "D$n has one wording"; else fail "D$n has $u different wordings"; fi
done

# --------------------------------------------------- 7. no paragraph in a row
head2 "7. no paragraph smuggled into a table row"
long=$(grep -nE '^\|' "$T" | awk -F: 'length($0) > 200 { print $1 }' | head -5)
if [ -n "$long" ]; then
  fail "table rows over 200 chars at lines: $(echo "$long" | tr '\n' ' ')"
else
  pass "none"
fi

# ------------------------------------------------- 8. named test files exist
head2 "8. every named test file exists"
missing=$(grep -E '^\|' "$T" \
  | awk -F'|' 'NF==6 { gsub(/[` ]/,"",$3); if ($3 ~ /\.swift$/) print $3 }' \
  | sort -u \
  | while read -r f; do
      [ -e "$REPO/VisionCapture/$f" ] || [ -e "$REPO/$f" ] || echo "$f"
    done)
if [ -n "$missing" ]; then
  fail "coverage table names test files that do not exist:"; printf '%s\n' "$missing" | sed 's/^/        /'
else
  pass "all present"
fi

# ------------------------------------------------------ 9. Now counters true
head2 "9. the Now counters are true"
real_cov=$(grep -E '^\|' "$T" | awk -F'|' 'NF==6 {
    gsub(/[` ]/,"",$4)
    if ($4 ~ /^-+$/ || $4 == "Result") next
    if ($4 !~ /^[0-9]+\/[0-9]+pass$/ && $4 !~ /^no-unit-test/ && $4 !~ /^waived/) c++
  } END { print c+0 }')
real_blk=$(grep -c '^- \[!\] ' "$T")
said_cov=$(grep -m1 'coverage rows not passing:' "$T" | sed 's/.*not passing: *`*\([0-9]*\)`*.*/\1/')
said_blk=$(grep -m1 '^- Blocked:'                "$T" | sed 's/^- Blocked: *`*\([0-9]*\)`*.*/\1/')
[ "${said_cov:-x}" = "$real_cov" ] \
  && pass "coverage rows not passing: $real_cov" \
  || fail "Now says '${said_cov:-none}' coverage rows not passing, the tables say $real_cov"
[ "${said_blk:-x}" = "$real_blk" ] \
  && pass "blocked: $real_blk" \
  || fail "Now says '${said_blk:-none}' blocked, the tasks say $real_blk"

# ------------------------------------------- 10. numbers spelled two ways
head2 "10. the same claim written with two numbers"
dupes=$(grep -ohE '[0-9]+ (errors?|sites?)' "$T" "$I" "$H" 2>/dev/null \
        | sort -u | awk '{print $2}' | sort | uniq -d)
if [ -n "$dupes" ]; then
  warn "one claim, two numbers - read these by eye:"
  for w in $dupes; do printf '        %s\n' "$(grep -ohE "[0-9]+ $w" $T $I $H 2>/dev/null | sort -u | tr '\n' ' ')"; done
else
  pass "no contradicting counts"
fi

# ------------------------------------------------- 11. the three files agree
head2 "11. the files agree"
if [ -n "$STRAY" ]; then
  fail "a page sits in the work-item folder: $STRAY - it belongs in $PF/human/ (ADR 0052)"
fi
if [ -n "$T" ] && [ -n "$I" ]; then
  pass "tracker and implementation document present"
else
  fail "missing one of tracker/implementation: T='$T' I='$I'"
fi
if [ ! -f "$H" ]; then
  fail "no owner page at $H - run scripts/build-page.py $PWD"
  H=""
else
  pass "owner page at human/$SLUG.html"
fi
if [ -n "$H" ]; then
  stamp=$(grep -m1 'generated-from:' "$H" 2>/dev/null)
  if [ -n "$stamp" ]; then
    want_t=$(echo "$stamp" | sed 's/.*tracker=\([0-9a-f]*\).*/\1/')
    want_i=$(echo "$stamp" | sed 's/.*implementation=\([0-9a-f]*\).*/\1/')
    have_t=$(shasum -a 1 "$T" | cut -c1-12)
    have_i=$(shasum -a 1 "$I" | cut -c1-12)
    if [ "$want_t" = "$have_t" ] && [ "$want_i" = "$have_i" ]; then
      pass "html stamp matches both markdown files"
    else
      fail "html was built from older markdown - regenerate it before handing over"
    fi
  else
    newest=$(ls -t "$T" "$I" "$H" | head -1)
    if [ "$newest" = "$H" ]; then
      pass "html is the newest file (no stamp - add one, see the template)"
    else
      fail "html is older than $newest - the owner is reading a stale page. Regenerate it."
    fi
  fi
fi

# ---------------------------------------------------- 12. approval is real
head2 "12. the approval is real"
status=$(grep -m1 '^status:' "$T" | awk '{print $2}')
if [ "$status" = "approved" ]; then
  contra=$(grep -inE 'unapproved|not approved|awaiting.approval' "$T" "$I" "$H" 2>/dev/null | head -3)
  [ -n "$contra" ] \
    && { fail "status: approved, but these lines disagree:"; printf '%s\n' "$contra" | sed 's/^/        /'; } \
    || pass "status approved, nothing contradicts it"
else
  pass "status: $status"
fi
claimed=$(grep -E '^approved_(by|at|version):' "$T" | grep -vc 'null')
if [ "$claimed" -gt 0 ]; then
  grep -q 'Owner said:' $I 2>/dev/null \
    && pass "approval fields are backed by a quoted owner line" \
    || fail "approval fields are set but $I has no 'Owner said:' quote - recipe 0a"
else
  pass "no approval claimed"
fi
ver=$(grep -m1 '^version:'          "$T" | awk '{print $2}')
apv=$(grep -m1 '^approved_version:' "$T" | awk '{print $2}')
[ "$ver" = "$apv" ] \
  && pass "approved_version $apv equals version $ver" \
  || warn "version $ver, approved_version $apv - work must stop until $ver is approved"

# ------------------------------------------------------------ 13. D4
head2 "13. D4 - no blind spot"
changed=$( cd "$REPO" && {
    [ -n "$BASE" ] && git diff --name-only "$BASE"..HEAD -- "${PATHS[@]}"
    git diff --name-only            -- "${PATHS[@]}"
    git diff --name-only --cached   -- "${PATHS[@]}"
    git ls-files --others --exclude-standard -- "${PATHS[@]}"
  } 2>/dev/null | sort -u )
dirty=$( cd "$REPO" && git status --porcelain -- "${PATHS[@]}" 2>/dev/null | head -1 )
if [ -z "$changed" ] && [ -n "$dirty" ]; then
  fail "D4 printed nothing while the tree has changes - THE COMMAND IS BROKEN, do not tick D4"
elif [ -z "$changed" ]; then
  pass "no source files changed"
else
  # First column of every coverage row. A row may name a file or a directory.
  # Take the first backticked token, so free text after it ("(ring and callback)")
  # does not corrupt the path.
  grep -E '^\|' "$T" | awk -F'|' 'NF==6 {
        if ($2 ~ /^ *-+ *$/ || $2 ~ /Code this phase/) next
        print $2
      }' \
    | sed -e 's/.*`\([^`]*\)`.*/\1/' -e 's/^ *//' -e 's/ *$//' \
    | grep -v '^$' | sort -u > "$CHUNKS/cov"

  absent=""
  for f in $changed; do
    short=${f#VisionCapture/}
    hit=no
    while IFS= read -r row; do
      # exact file row, or a directory row that is a prefix of this path
      [ "$row" = "$short" ] && { hit=yes; break; }
      case "$row" in
        */) case "$short" in "$row"*) hit=yes; break ;; esac ;;
      esac
    done < "$CHUNKS/cov"
    [ "$hit" = no ] && absent="$absent$f"$'\n'
  done

  if [ -n "$absent" ]; then
    fail "changed but absent from every coverage table:"
    printf '%s' "$absent" | sed 's/^/        /'
  else
    pass "$(echo "$changed" | wc -l | tr -d ' ') changed files, all covered by a row or a directory row"
  fi
fi
[ -n "$BASE" ] || warn "no --base given - D4 covered the working tree only, not commits since the phase started"

# ---------------------------------------- 14. a done phase carries no blocker
head2 "14. no stale 'Blocked by:' line"
bad=""; soft=""
for f in "$CHUNKS"/p*; do
  [ -e "$f" ] || continue
  grep -q '^Blocked by:' "$f" || continue
  name=$(head -1 "$f")
  if grep -q '^Status: `\[x\]`' "$f"; then
    bad="$bad$name"$'\n'
  elif ! grep -q '^- \[!\] ' "$f" && ! grep -E '^\|' "$f" | awk -F'|' 'NF==6 {
        gsub(/[` ]/,"",$4); if ($4 ~ /^-+$/ || $4 == "Result") next
        if ($4 !~ /^[0-9]+\/[0-9]+pass$/ && $4 !~ /^no-unit-test/ && $4 !~ /^waived/) f=1
      } END { exit !f }'; then
    soft="$soft$name"$'\n'
  fi
done
[ -n "$bad" ]  && { fail "a closed phase still says it is blocked:"; printf '%s\n' "$bad" | sed 's/^/        /'; }
[ -n "$soft" ] && { warn "'Blocked by:' but nothing is stuck - delete the line (recipe 4):"
                    printf '%s\n' "$soft" | sed 's/^/        /'; }
[ -z "$bad$soft" ] && pass "none"

# ------------------------------------------------ 15. a done phase is really done
head2 "15. D1 - a closed phase holds no open task"
bad=""
for f in "$CHUNKS"/p*; do
  [ -e "$f" ] || continue
  grep -q '^Status: `\[x\]`' "$f" || continue
  open=$(grep -cE '^- \[[~!s]\] [0-9]+\.[0-9]+ ' "$f")
  [ "$open" -gt 0 ] && bad="$bad$(head -1 "$f") - $open task(s) still [~], [!] or [s]"$'\n'
done
if [ -n "$bad" ]; then
  fail "D1 is ticked but tasks are open:"; printf '%s\n' "$bad" | sed 's/^/        /'
else
  pass "none"
fi
s_count=$(grep -c '^- \[s\] ' "$T")
[ "$s_count" -gt 0 ] && warn "$s_count task(s) are [s] source-done - code accepted, proof not passed. They do not close a phase."

# --------------------------------------------------- 16. waived is the owner's
head2 "16. every 'waived' row is the owner's decision"
w=$(grep -E '^\|' "$T" | awk -F'|' 'NF==6 { gsub(/^[ `]+|[ `]+$/,"",$4); if ($4 ~ /^waived/) print $4 }')
n_w=$(printf '%s' "$w" | grep -c . )
if [ "$n_w" -eq 0 ]; then
  pass "no waived rows"
else
  # The approval quote is not a waiver quote. Each waiver needs its own row and
  # its own quoted owner sentence, in the ## Waivers section. Recipe 0b.
  awk '/^## Waivers/{f=1;next} /^## /{f=0} f' "$I" > "$CHUNKS/waivers" 2>/dev/null
  n_rows=$(awk -F'|' 'NF>=4 && $0 !~ /^\| *-+/ && $0 !~ /\| *Date *\|/ {c++} END{print c+0}' "$CHUNKS/waivers")
  n_q=$(grep -c 'Owner said:' "$CHUNKS/waivers")
  if [ ! -s "$CHUNKS/waivers" ]; then
    fail "$n_w waived row(s) but $I has no '## Waivers' section - only the owner waives a test (recipe 0b):"
    printf '%s\n' "$w" | sed 's/^/        /'
  elif [ "$n_rows" -lt "$n_w" ] || [ "$n_q" -lt "$n_w" ]; then
    fail "$n_w waived row(s), but ## Waivers has $n_rows row(s) and $n_q quoted owner line(s) - each waiver needs both"
  else
    pass "$n_w waived row(s), each with a row and an owner quote in ## Waivers"
  fi
fi

# ------------------------------------------- 17. no-unit-test names real proof
head2 "17. every 'no-unit-test' names a real proof"
bad=""; vague=""
grep -E '^\|' "$T" | awk -F'|' 'NF==6 { gsub(/^[ `]+|[ `]+$/,"",$4); if ($4 ~ /^no-unit-test/) print $4 }' \
| while IFS= read -r cell; do
    proof=${cell#no-unit-test - }
    # A path token must start with a letter. "20/20 pass" is a count, not a path.
    p=$(echo "$proof" | grep -oE '[A-Za-z][A-Za-z0-9_.-]*/[A-Za-z0-9_./-]*' | head -1)
    if [ -n "$p" ]; then
      if [ -e "$p" ] || [ -e "$REPO/$p" ] || [ -e "$REPO/VisionCapture/$p" ]; then
        echo "OK|"
      else
        echo "BAD|$p"
      fi
    else
      echo "VAGUE|$proof"
    fi
  done > "$CHUNKS/nut"
bad=$(grep '^BAD|'   "$CHUNKS/nut" | cut -d'|' -f2)
vague=$(grep '^VAGUE|' "$CHUNKS/nut" | cut -d'|' -f2)
[ -n "$bad" ]   && { fail "proof path does not exist:";  printf '%s\n' "$bad"   | sed 's/^/        /'; }
[ -n "$vague" ] && { warn "proof names no evidence path - say where the run is recorded:"
                    printf '%s\n' "$vague" | sed 's/^/        /'; }
[ -z "$bad$vague" ] && pass "all proofs point at something real"

# ---------------------------------------------------------------- result
head2 "----"
if [ "$fails" -gt 0 ]; then
  printf '%s FAIL, %s WARN. Do not hand over.\n' "$fails" "$warns"
  exit 1
fi
printf 'All checks passed. %s WARN.\n' "$warns"
exit 0
