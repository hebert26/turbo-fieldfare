#!/usr/bin/env python3
"""Build the owner's HTML page from the tracker and the implementation document.

    build-page.py <work-item-folder> [--out FILE]

With no --out it rewrites the page in place. Run it every time either Markdown
file changes - recipe 2 and recipe 7 in references/update-recipes.md.

WHAT IT GENERATES, from the two Markdown files, every run:

  phase entries   task lines, acceptance boxes, coverage counts and the D1-D5
                  marks - all copied or computed, never retold in prose
  waits-on line   what each phase needs, what it blocks, what it can run beside
  cleanup rows    one clean at the end of the last phase; every other phase says
                  so and points at it
  prompts         one runnable prompt per open task, for phases that can start
                  today. Finished tasks get none.
  stats and chips the four counters and the four filter counts
  stamp           a sha of both Markdown files, so self-check 11 can prove the
                  page is not stale

WHAT IT NEVER TOUCHES - these are human judgment and are kept word for word:

  the header, the goal sentence, the approved-scope card, each phase's one-line
  lede, and every t-dev / t-disc / t-human entry.

It is safe to run twice. Everything it writes sits between BUILD markers, or is
replaced by matching on its own output.
"""
import argparse
import hashlib
import os
import html
import re
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("folder", help="the work-item folder")
ap.add_argument("--out", help="write here instead of rewriting the page in place")
args = ap.parse_args()

FOLDER = Path(args.folder)


def one(pattern, what):
    hits = sorted(FOLDER.glob(pattern))
    if len(hits) != 1:
        sys.exit(f"expected exactly one {what} in {FOLDER}, found {len(hits)}")
    return hits[0]


tracker_p = one("tracker-*.md", "tracker")
impl_p = one("implementation-*.md", "implementation document")
tracker = tracker_p.read_text()
impl_md = impl_p.read_text()

# The owner's page does NOT live in the work-item folder. It lives in
# project-files/human/, so an agent listing the work item never sees it and
# never spends context on it. ADR 0052. Nothing points at it from either
# Markdown file - the path is derived from the slug, both here and by hand.
slug_m = re.search(r"^slug:\s*(\S+)\s*$", tracker, re.M)
slug = slug_m.group(1).strip().strip('"') if slug_m else FOLDER.resolve().name

root = FOLDER.resolve()
while root.parent != root and root.name != "project-files":
    root = root.parent
if root.name != "project-files":
    sys.exit(f"{FOLDER} is not inside a project-files/ tree - cannot place the owner page")
HUMAN = root / "human"
HUMAN.mkdir(parents=True, exist_ok=True)
page_p = HUMAN / f"{slug}.html"

TEMPLATE = Path(__file__).resolve().parent.parent / "references" / "implementation-plan-template.html"
stray = sorted(FOLDER.glob("implementation-*.html"))
if not page_p.exists():
    if stray and args.out:
        # a preview run must never move files. Read the old one where it sits.
        page_p = stray[0]
    elif stray:
        # migrating a work item written before ADR 0052
        page_p.write_text(stray[0].read_text())
        print(f"moved {stray[0].name} -> {page_p}")
        stray[0].unlink()
    elif TEMPLATE.exists():
        page_p.write_text(TEMPLATE.read_text())
        print(f"seeded {page_p} from the template")
    else:
        sys.exit(f"no page at {page_p} and no template at {TEMPLATE}")
elif stray:
    sys.exit(f"a page still sits in the work-item folder: {stray[0]}\n"
             f"The real one is {page_p}. Delete the stray copy - two pages means one lies.")

OUT = Path(args.out) if args.out else page_p
page = page_p.read_text()

# ---------------------------------------------------------------- parse tracker
phases = []
blocks = re.split(r"^## Phase ", tracker, flags=re.M)[1:]
for b in blocks:
    b = re.split(r"^## ", b, flags=re.M)[0]
    head, _, rest = b.partition("\n")
    num, _, name = head.partition(" - ")
    p = {"n": int(num.strip()), "name": name.strip()}

    strip = re.search(r"^Status: `\[(.)\]` - Owner: `([^`]*)` - Needs: `([^`]*)`", rest, re.M)
    p["status"] = strip.group(1) if strip else " "
    p["owner"] = strip.group(2) if strip else "-"
    p["needs"] = strip.group(3) if strip else "-"

    p["tasks"] = [
        {"mark": m.group(1), "id": m.group(2), "text": m.group(3).strip(), "date": (m.group(4) or "").strip()}
        for m in re.finditer(r"^- \[(.)\] (\d+\.\d+[A-Za-z]?) (.*?)(?:[ \t]{2,}(\S+))?$", rest, re.M)
    ]
    # acceptance = ticked lines that are not tasks and not D-lines
    acc_block = re.search(r"\*\*Acceptance\*\*\n(.*?)\n\*\*Unit test coverage\*\*", rest, re.S)
    p["acceptance"] = (
        [(m.group(1), m.group(2).strip()) for m in re.finditer(r"^- \[(.)\] (.*)$", acc_block.group(1), re.M)]
        if acc_block else []
    )
    cov_block = re.search(r"\*\*Unit test coverage\*\*\n(.*?)\n\*\*Done when\*\*", rest, re.S)
    rows = []
    if cov_block:
        for line in cov_block.group(1).splitlines():
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) == 4 and not cells[0].startswith("-") and cells[0] != "Code this phase changed":
                rows.append(cells)
    p["cov"] = rows
    p["blocked"] = (re.search(r"^Blocked by: (.*)$", rest, re.M) or [None, ""])[1] if re.search(r"^Blocked by:", rest, re.M) else ""
    phases.append(p)

PASSING = re.compile(r"^`?(\d+)/\1 pass|^`?no-unit-test|^`?waived")


def gates(p):
    """D1..D5 computed from the tracker. Returns (state, reason) per gate."""
    open_tasks = [t for t in p["tasks"] if t["mark"] in "~! s".strip() or t["mark"] in ("~", "!", "s", " ")]
    d1 = not open_tasks
    unticked = [a for a in p["acceptance"] if a[0] != "x"]
    d2 = bool(p["acceptance"]) and not unticked
    bad = [r for r in p["cov"] if not PASSING.match(r[2])]
    d3 = bool(p["cov"]) and not bad
    return (
        [d1, d2, d3, None, None],
        {
            "open": len(open_tasks),
            "unticked": len(unticked),
            "bad": len(bad),
            "cov_total": len(p["cov"]),
        },
    )


def blocked_by(n):
    return [q["n"] for q in phases if str(n) in [x.strip() for x in q["needs"].split(",")]]


MARK_LABEL = {"x": "done", "~": "in progress", " ": "not started", "!": "blocked", "-": "skipped", "s": "source-done"}

def esc(s):
    return html.escape(s, quote=False)


# The one cleanup task: the last task of the last phase. Find it, do not assume 7.7.
cleanup_id = "?"
for q in phases:
    for t in q["tasks"]:
        if re.match(r"clean\s*up\b", t["text"], re.I):
            cleanup_id = t["id"]
if cleanup_id != "?" and cleanup_id.split(".")[0] != str(phases[-1]["n"]):
    print(f"WARNING: cleanup task {cleanup_id} is not in the last phase", file=sys.stderr)

entries = []
for p in phases:
    st, cnt = gates(p)
    running = p["status"] in ("~",)

    pips = []
    for i, ok in enumerate(st, start=1):
        if ok is True:
            cls, gl = "ok", "&#10003;"
        elif ok is False and running:
            cls, gl = "bad", "&#10007;"
        elif ok is False:
            cls, gl = "open", "&middot;"
        else:
            cls, gl = "unknown", "?"
        pips.append(f'<span class="pip {cls}">D{i}<i>{gl}</i></span>')
    pipbar = '<span class="pips">' + "".join(pips) + "</span>"
    pipbar_mini = '<span class="pips mini-pips" aria-hidden="true">' + "".join(
        f'<span class="dot {("ok" if o is True else ("bad" if (o is False and running) else ("unknown" if o is None else "open")))}"></span>'
        for o in st
    ) + "</span>"

    why = []
    if cnt["open"]:
        why.append(f"D1 &mdash; {cnt['open']} task{'s' if cnt['open'] != 1 else ''} still open")
    if cnt["unticked"]:
        why.append(f"D2 &mdash; {cnt['unticked']} acceptance box{'es' if cnt['unticked'] != 1 else ''} unticked")
    if cnt["bad"]:
        why.append(f"D3 &mdash; {cnt['bad']} of {cnt['cov_total']} coverage rows not passing")
    why.append("D4 and D5 not checked yet")
    whytxt = ". ".join(why) + "."

    # --- tasks, copied
    rows = []
    for t in p["tasks"]:
        m = t["mark"]
        glyph = {"x": "&#10003;", "~": "&#9679;", " ": "&nbsp;", "-": "&ndash;", "!": "!", "s": "s"}.get(m, "?")
        date = t["date"]
        if m in ("x", "-") and (not date or date == "unknown"):
            dcell = '<em class="nodate">date unknown</em>'
        elif date and date != "-":
            dcell = f'<span class="d">{esc(date)}</span>'
        else:
            dcell = ""
        cls = {"x": "ok", "~": "doing", "-": "skip", "!": "blocked", "s": "src"}.get(m, "todo")
        text = esc(t["text"])
        # the one real cleanup task
        if t["id"] == cleanup_id:
            cls += " is-cleanup"
            text += ' <span class="tag">gives the disk back &middot; closes the work item</span>'
        # a phase that only points at it
        elif re.search(r"cleanup moved to", t["text"], re.I):
            cls += " is-pointer"
            text = f'Cleanup happens once, at the end <span class="tag ghost">&rarr; task {cleanup_id}</span>'
            dcell = ""
        # the date rides inside the text cell, so a wrapped line can never read
        # as if it belonged to the task below it
        rows.append(
            f'<li class="m-{cls}"><span class="bx">{glyph}</span>'
            f'<span class="id">{t["id"]}</span><span class="tx">{text}{dcell}</span></li>'
        )
    tasklist = '<ul class="tasklist">' + "".join(rows) + "</ul>"

    done = sum(1 for t in p["tasks"] if t["mark"] in ("x", "-"))

    # --- acceptance, copied
    acc = "".join(
        f'<li class="{"ok" if a[0]=="x" else "todo"}"><span class="bx">{"&#10003;" if a[0]=="x" else "&nbsp;"}</span>{esc(a[1])}</li>'
        for a in p["acceptance"]
    )
    acclist = f'<ul class="acclist">{acc}</ul>'

    # --- coverage summary
    good = cnt["cov_total"] - cnt["bad"]
    badnames = [r[0].replace("`", "").strip() for r in p["cov"] if not PASSING.match(r[2])][:3]
    more = cnt["bad"] - len(badnames)
    covtxt = f'<strong>{good} of {cnt["cov_total"]}</strong> coverage rows pass.'
    if badnames:
        covtxt += " Waiting on " + ", ".join(f"<code>{esc(b)}</code>" for b in badnames)
        covtxt += f" and {more} more." if more > 0 else "."

    # --- waits-on line
    needs = [x.strip() for x in p["needs"].split(",") if x.strip() and x.strip() != "-"]
    blocks_ = blocked_by(p["n"])
    bits = []
    bits.append("waits on " + ", ".join(f"P{x}" for x in needs) if needs else "waits on nothing")
    bits.append("blocks " + ", ".join(f"P{x}" for x in blocks_) if blocks_ else "blocks nothing")
    siblings = [q["n"] for q in phases if q["n"] != p["n"] and q["needs"] == p["needs"] and needs]
    if siblings:
        bits.append("can run beside " + ", ".join(f"P{x}" for x in siblings))
    waits = " &middot; ".join(bits)

    # --- reuse the human lede from the existing page
    #
    # Cut the phase's own entry block out FIRST, then look inside it. A regex that
    # searches the whole document happily runs past the end of one entry and takes
    # a paragraph from a later one - on the second build every phase ended up with
    # the same discovery text. Never let this pattern span entries.
    lede = ""
    blk = re.search(r'<span class="badge">Phase %d &middot;.*?(?=<div class="entry |\Z)' % p["n"],
                    page, re.S)
    if blk:
        pm = re.search(r"<p>(.*?)</p>", blk.group(0), re.S)
        if pm:
            lede = pm.group(1).strip()

    is_last = p["n"] == phases[-1]["n"]
    lastbadge = ' <span class="finalchip">final phase &middot; closes the work item</span>' if is_last else ""

    # cleanup row: one clean, at the end of the last phase
    if is_last:
        cleanup_row = (
            '<div class="row cleanup"><div class="k">Cleanup</div><div class="v">'
            f'<strong>One clean, here, at the end.</strong> Task {cleanup_id} is the only cleanup task in the '
            'whole work item, and D1 on this phase blocks until it is done &mdash; so the work item cannot '
            'close until the disk is given back. Phases 1&ndash;6 record a disk baseline when they start and '
            'nothing else; those baselines are what tell this task what the work item created from what was '
            'already on the machine.</div></div>'
        )
    else:
        cleanup_row = (
            '<div class="row cleanup calm"><div class="k">Cleanup</div><div class="v">'
            f'None here, by design. Baseline recorded at phase start; the one clean runs at task {cleanup_id}, '
            'at the end of the last phase.</div></div>'
        )

    entries.append(f'''
      <div class="entry t-plan{' is-last' if is_last else ''}" data-t="plan" id="p{p['n']}" data-jump="P{p['n']}" data-state="{'done' if p['status']=='x' else ('doing' if p['status']=='~' else 'todo')}">
        <div class="ts">P{p['n']}</div>
        <div class="rail"><span class="node"></span></div>
        <div class="body">
          <details class="fold"{' open' if p['status'] in ('~', 'x') or is_last else ''}>
            <summary>
              <span class="badge">Phase {p['n']} &middot; {MARK_LABEL.get(p['status'], '?')}</span><span class="ts-mobile">P{p['n']}</span>{lastbadge}
              <h3>{esc(p['name'])}</h3>
              <span class="waits">{waits} &middot; owner {esc(p['owner'])}</span>
              <span class="mini">{pipbar_mini}<span class="minitask">{done}/{len(p['tasks'])} tasks</span></span>
              <span class="chev" aria-hidden="true"></span>
            </summary>
            <div class="foldbody">
              <p>{lede}</p>
              <div class="devgrid">
                <div class="row"><div class="k">Tasks &middot; {done} of {len(p['tasks'])}</div><div class="v">{tasklist}</div></div>
                <div class="row"><div class="k">Acceptance</div><div class="v">{acclist}</div></div>
                <div class="row{' chosen' if cnt['bad'] == 0 and cnt['cov_total'] else ''}"><div class="k">Unit tests</div><div class="v">{covtxt}</div></div>
                {cleanup_row}
                <div class="row"><div class="k">Done when</div><div class="v">{pipbar}<div class="why">{whytxt}</div>{'<div class="why blk">Blocked by: ' + esc(p['blocked']) + '</div>' if p['blocked'] else ''}</div></div>
              </div>
            </div>
          </details>
        </div>
      </div>
''')

# ------------------------------------------------------------------ prompts
# One runnable prompt per OPEN task, but only for phases that can actually
# start today. A finished task needs no prompt, and a phase that is still
# waiting on another phase cannot be worked on yet - generating those would
# just make the page heavier for no one.
impl_name = next(FOLDER.glob("implementation-*.md")).name

OPEN_MARKS = (" ", "~", "!", "s")


def task_detail(tid):
    """Pull one task's real detail out of the implementation document."""
    m = re.search(r"^#### " + re.escape(tid) + r"\b(.*?)(?=^#### |^### |^## |\Z)",
                  impl_md, re.S | re.M)
    if not m:
        return None
    body = m.group(1)
    grab = lambda k: (re.search(r"\|\s*" + k + r"\s*\|([^|]*)\|", body) or [None, ""])[1].strip()
    acc_m = re.search(r"\*\*Acceptance detail\*\*\n(.*?)(?=\n\*\*|\Z)", body, re.S)
    acc = re.findall(r"^- \[.\] (.*)$", acc_m.group(1), re.M) if acc_m else []
    chk_m = re.search(r"\*\*How to check\*\*\s*\n+```[a-z]*\n(.*?)```", body, re.S)
    chk = chk_m.group(1).strip() if chk_m else ""
    # the prose sits between the meta table and the first bold block
    prose = re.split(r"\n\*\*", body)[0]
    prose = "\n".join(prose.splitlines()[1:])                # drop the rest of the #### heading
    prose = re.sub(r"^\s*\|.*$", "", prose, flags=re.M)      # drop the meta table
    # the source is hard-wrapped; rejoin each paragraph so it can be re-flowed cleanly
    paras = [" ".join(x.split()) for x in re.split(r"\n\s*\n", prose) if x.strip()]
    prose = "\n\n".join(paras)
    return {"touches": grab("Touches"), "depends": grab("Depends on"),
            "parallel": grab("Parallel safe"), "acc": acc, "check": chk, "prose": prose}


def wrap(label, text, width=76, nowrap=False):
    """LABEL  first line, continuation lines aligned under it.

    nowrap=True keeps every line whole. Shell commands MUST use it - a wrapped
    command is a broken command the moment someone pastes it.
    """
    pad = " " * 11
    if nowrap:
        lines = [l for l in str(text).split("\n")]
        return "\n".join((f"{label:<11}" if i == 0 else pad) + l for i, l in enumerate(lines))
    out, first = [], True
    for para in str(text).split("\n"):
        if not para.strip():
            out.append("")
            continue
        line = ""
        for word in para.split():
            if line and len(line) + 1 + len(word) > width:
                out.append((f"{label:<11}" if first else pad) + line)
                first, line = False, word
            else:
                line = f"{line} {word}".strip()
        if line:
            out.append((f"{label:<11}" if first else pad) + line)
            first = False
    if first:
        out.append(f"{label:<11}")
    return "\n".join(out)


def task_prompt(p, t):
    d = task_detail(t["id"])
    if not d:
        return None
    L = [wrap("GOAL", f'{t["text"]} (task {t["id"]}, phase {p["n"]}).')]
    L.append(wrap("SOURCE", f'{impl_name} -> section "#### {t["id"]}". '
                            f'Open it only if you need more than this prompt gives.'))
    if d["touches"]:
        L.append(wrap("TOUCHES", f'{d["touches"].replace("`","")} - change nothing else.'))
    dep = d["depends"] or "nothing"
    L.append(wrap("DEPENDS", f'{dep} - parallel safe: {d["parallel"] or "unknown"}.'))
    if d["prose"]:
        L.append(wrap("CONTEXT", d["prose"]))
    if d["acc"]:
        L.append(wrap("DONE WHEN", "- " + d["acc"][0]))
        for a in d["acc"][1:]:
            L.append(wrap("", "- " + a))
    if d["check"]:
        L.append(wrap("VERIFY", d["check"], nowrap=True))
        L.append(wrap("", 'A pass needs "Executed N tests" with N above zero and zero '
                          'failures. A clean exit that ran nothing is a FAIL.'))
    L.append(wrap("EVIDENCE", f'Write the real numbers into {impl_name} -> '
                              f'"Phase {p["n"]} evidence". Never write "done" without saying how you know.'))
    L.append(wrap("RULES", "Follow AGENTS.md. Do not commit or push - commits belong to Hebert. "
                           "If this prompt and the markdown disagree, the markdown wins."))
    return "\n".join(L)


def phase_prompt(p, open_tasks):
    L = [wrap("GOAL", f'Phase {p["n"]}: {p["name"]}. Finish the open tasks so this phase can close.')]
    L.append(wrap("SOURCE", f'{impl_name} -> "## Phase {p["n"]}". Read that section first.'))
    L.append(wrap("OPEN", f'{open_tasks[0]["id"]} {open_tasks[0]["text"]}'))
    for t in open_tasks[1:]:
        L.append(wrap("", f'{t["id"]} {t["text"]}'))
    unticked = [a[1] for a in p["acceptance"] if a[0] != "x"]
    if unticked:
        L.append(wrap("DONE WHEN", "- " + unticked[0]))
        for a in unticked[1:]:
            L.append(wrap("", "- " + a))
    L.append(wrap("GATE", "The phase closes only when D1-D5 all pass. D3 and D4 are the test gate: "
                          "every coverage row passes, and no file this phase changed is missing "
                          "from the coverage table."))
    L.append(wrap("RULES", "Follow AGENTS.md. One task at a time. Do not commit or push."))
    return "\n".join(L)


def startable(p):
    if p["status"] == "~":
        return True
    needs = [x.strip() for x in p["needs"].split(",") if x.strip() and x.strip() != "-"]
    if not needs:
        return p["status"] != "x"
    by_n = {q["n"]: q for q in phases}
    return all(by_n.get(int(n), {}).get("status") == "x" for n in needs if n.isdigit())


prompt_blocks, n_prompts = [], 0
waiting = []
for p in phases:
    open_tasks = [t for t in p["tasks"] if t["mark"] in OPEN_MARKS]
    if not open_tasks:
        continue
    if not startable(p):
        waiting.append(p)
        continue
    items = []
    pp = phase_prompt(p, open_tasks)
    n_prompts += 1
    items.append(
        '<details class="pcard whole"><summary><span class="ptag">whole phase</span>'
        f'<span class="pname">Phase {p["n"]} &middot; {esc(p["name"])}</span>'
        '<span class="chev" aria-hidden="true"></span></summary>'
        f'<div class="pwrap"><button class="copyp">copy</button><pre>{esc(pp)}</pre></div></details>'
    )
    for t in open_tasks:
        tp = task_prompt(p, t)
        if not tp:
            continue
        n_prompts += 1
        mark = "in progress" if t["mark"] == "~" else ("blocked" if t["mark"] == "!" else "not started")
        items.append(
            f'<details class="pcard"><summary><span class="ptag t">{t["id"]}</span>'
            f'<span class="pname">{esc(t["text"])}</span>'
            f'<span class="pstate">{mark}</span>'
            '<span class="chev" aria-hidden="true"></span></summary>'
            f'<div class="pwrap"><button class="copyp">copy</button><pre>{esc(tp)}</pre></div></details>'
        )
    prompt_blocks.append(
        f'<div class="pgroup"><h3>Phase {p["n"]} &mdash; {esc(p["name"])}</h3>'
        + "".join(items) + "</div>"
    )

wait_note = ""
if waiting:
    rows = ", ".join(f'P{p["n"]} (waits on P{p["needs"]})' for p in waiting)
    wait_note = (f'<p class="sub">No prompts yet for {rows}. '
                 f'A phase you cannot start today does not need one &mdash; they appear here '
                 f'the moment the phase it waits on closes.</p>')

PROMPTS = f'''
  <div class="divider">Run it</div>
  <div class="doc" id="prompts">
    <div class="doc-head">
      <span class="dot"></span>
      <span class="path">Prompts you can run now</span>
      <span class="meta">{n_prompts} ready &middot; generated from the tracker</span>
    </div>
    <div class="pbody">
      <p class="sub">Each prompt is self-contained: the task detail is inlined, so the model does
      not have to open a {len(impl_md)//1000} KB document to find one section. Copy one, paste it
      into Codex or Claude Code, and it has the goal, the boundary, what done means, and the exact
      command that proves it.</p>
      {wait_note}
      {"".join(prompt_blocks)}
      <p class="sub foot">Finished tasks get no prompt. These are regenerated from the tracker every
      time this page is built &mdash; never edit one by hand. If a prompt and the markdown disagree,
      the markdown wins.</p>
    </div>
  </div>
'''

# ------------------------------------------------- splice into the existing page
# Everything generated lives between markers, so a second run replaces its own
# output instead of stacking a copy on top of it.
A, B = "<!-- BUILD:phases -->", "<!-- /BUILD:phases -->"
block = A + "\n" + "".join(entries) + "\n      " + B
if A in page and B in page:
    new = page[: page.index(A)] + block + page[page.index(B) + len(B):]
else:
    # first build on a page written before the builder existed
    open_tag = '<div class="timeline" id="timeline">'
    if open_tag not in page:
        sys.exit("no <div class=\"timeline\" id=\"timeline\"> in the page - copy the template first")
    start = page.index(open_tag) + len(open_tag)
    ends = [page.find(f'<div class="entry t-{k}"') for k in ("dev", "disc", "human")]
    ends = [e for e in ends if e > start]
    if not ends:
        sys.exit("cannot find where the phase entries stop. Add the BUILD:phases markers "
                 "to the page by hand, once, then rerun.")
    new = page[:start] + "\n" + block + "\n      " + page[min(ends):]

EXTRA_CSS = """
  /* ── task-tracker build additions ── */
  /* ── changes 1-4 ── */
  .waits { font-family: var(--mono); font-size: 11px; letter-spacing: 0.04em; color: var(--g500); margin: 0 0 8px; }
  .tasklist, .acclist { list-style: none; margin: 0; padding: 0; display: grid; gap: 10px; }
  /* a list row is a tall block, not a one-line key/value - give it more room than the
     short rows above and below it, or the first and last task touch the card edge */
  .devgrid .row:has(.tasklist), .devgrid .row:has(.acclist) { padding-top: 16px; padding-bottom: 18px; }
  .tasklist li, .acclist li {
    display: grid; grid-template-columns: 18px auto 1fr; gap: 8px;
    align-items: baseline; font-size: 13px; line-height: 1.45;
  }
  .acclist li { grid-template-columns: 18px 1fr; }
  .tasklist .tx, .acclist li { min-width: 0; overflow-wrap: anywhere; }
  .tasklist .d, .tasklist .nodate { margin-left: 6px; }
  .tasklist .bx, .acclist .bx {
    font-family: var(--mono); font-size: 10px; text-align: center;
    border: 1.5px solid var(--g300); border-radius: 4px; background: var(--paper);
    color: var(--g500); padding: 1px 0; line-height: 1.3;
  }
  .tasklist .id { font-family: var(--mono); font-size: 11px; color: var(--g500); font-variant-numeric: tabular-nums; }
  .tasklist .tx { min-width: 0; }
  .tasklist .d { font-family: var(--mono); font-size: 10.5px; color: var(--g500); white-space: nowrap; }
  .tasklist .nodate {
    font-family: var(--mono); font-size: 10px; font-style: normal; white-space: nowrap;
    color: var(--clay-d); border: 1px dashed var(--clay); border-radius: 4px; padding: 0 5px;
  }
  .tasklist .m-ok .bx  { border-color: var(--olive); background: #EEF1E8; color: var(--olive); }
  .tasklist .m-ok .tx  { color: var(--g500); }
  .tasklist .m-doing .bx  { border-color: var(--clay); background: #FBEDE6; color: var(--clay-d); }
  .tasklist .m-doing .tx  { color: var(--slate); font-weight: 600; }
  .tasklist .m-skip .bx  { border-style: dashed; }
  .tasklist .m-skip .tx, .tasklist .m-skip .d { color: var(--g500); text-decoration: line-through; text-decoration-color: var(--g300); }
  .acclist .ok .bx { border-color: var(--olive); background: #EEF1E8; color: var(--olive); }
  .acclist .ok { color: var(--g500); }

  .pips { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 8px; }
  .pip {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.06em;
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 7px; border-radius: 5px;
    border: 1.5px solid var(--g300); background: var(--paper); color: var(--g500);
  }
  .pip i { font-style: normal; font-size: 11px; line-height: 1; }
  .pip.ok      { border-color: var(--olive);  background: #EEF1E8; color: var(--olive); }
  .pip.bad     { border-color: var(--clay-d); background: #FBEDE6; color: var(--clay-d); }
  .pip.unknown { border-style: dashed; }
  .legend {
    display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px;
    padding: 14px 22px 0; font-family: var(--mono); font-size: 10.5px;
    letter-spacing: 0.06em; color: var(--g500);
  }
  .legend .lgt { text-transform: uppercase; letter-spacing: 0.12em; margin-right: 4px; }
  .legend .lg { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
  .legend .pip { padding: 2px 5px; }
  .why { font-size: 12.5px; color: var(--g700); }
  .why.blk { margin-top: 4px; color: var(--clay-d); }

  /* ── cleanup ── */
  .finalchip {
    display: inline-block; font-family: var(--mono); font-size: 9.5px;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 2.5px 8px; border-radius: 6px; margin-bottom: 6px;
    color: var(--clay-d); background: #FBEDE6; border: 1px solid var(--clay);
  }
  .row.cleanup .k { color: var(--clay-d); }
  .row.cleanup.calm { background: rgba(255,255,255,.45); }
  .row.cleanup.calm .k, .row.cleanup.calm .v { color: var(--g500); }
  .tasklist .tag {
    font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.06em;
    text-transform: uppercase; white-space: nowrap;
    padding: 1px 6px; border-radius: 5px; margin-left: 4px;
    color: var(--clay-d); background: #FBEDE6; border: 1px solid var(--oat);
  }
  .tasklist .tag.ghost { color: var(--g500); background: var(--g100); border-color: var(--g200); }
  .tasklist .is-cleanup .bx { border-color: var(--clay); color: var(--clay-d); }
  .tasklist .is-cleanup .tx { color: var(--slate); }
  .tasklist .is-pointer .tx { text-decoration: none; color: var(--g500); }
  .tasklist .is-pointer .bx { border-style: dotted; }

  /* ── collapsible phases ── */
  details.fold > summary {
    list-style: none; cursor: pointer; display: block;
    padding: 8px 34px 8px 0; margin: -8px 0 0; position: relative;
    border-radius: 8px;
  }
  details.fold > summary::-webkit-details-marker { display: none; }
  details.fold > summary:hover { background: rgba(0,0,0,.02); }
  details.fold > summary:focus-visible { outline: 2px solid var(--clay); outline-offset: 2px; }
  details.fold > summary h3 { margin: 2px 0 3px; }
  details.fold > summary .waits { display: block; }
  .chev {
    position: absolute; right: 6px; top: 12px;
    width: 9px; height: 9px; border-right: 1.8px solid var(--g500); border-bottom: 1.8px solid var(--g500);
    transform: rotate(45deg); transition: transform .15s ease;
  }
  details[open] > summary .chev { transform: rotate(-135deg); }
  .mini { display: none; align-items: center; gap: 8px; margin-top: 5px; }
  details:not([open]) > summary .mini { display: flex; }
  .mini-pips { gap: 3px; }
  .dot { width: 7px; height: 7px; border-radius: 2px; border: 1.5px solid var(--g300); background: var(--paper); }
  .dot.ok  { border-color: var(--olive);  background: var(--olive); }
  .dot.bad { border-color: var(--clay-d); background: var(--clay); }
  .dot.unknown { border-style: dotted; }
  .minitask { font-family: var(--mono); font-size: 10.5px; color: var(--g500); }
  .foldbody { padding-top: 2px; }

  /* ── bottom jump bar ── */
  .jump {
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 40;
    display: flex; justify-content: center;
    padding: 8px 10px calc(8px + env(safe-area-inset-bottom));
    background: linear-gradient(to top, var(--ivory) 62%, rgba(250,249,245,0));
    pointer-events: none;
  }
  .jump .bar {
    pointer-events: auto;
    display: flex; align-items: center; gap: 4px;
    padding: 5px; border-radius: 999px;
    background: var(--paper); border: 1.5px solid var(--g300);
    box-shadow: 0 4px 16px rgba(20,20,19,.10);
    max-width: 100%; overflow-x: auto; scrollbar-width: none;
  }
  .jump .bar::-webkit-scrollbar { display: none; }
  .jump a, .jump button {
    flex: none; display: inline-flex; align-items: center; justify-content: center;
    min-width: 38px; height: 38px; padding: 0 10px;
    border-radius: 999px; border: 1.5px solid transparent; background: transparent;
    font-family: var(--mono); font-size: 13px; color: var(--g700);
    text-decoration: none; cursor: pointer; position: relative;
  }
  .jump a::after {
    content: ""; position: absolute; bottom: 5px; left: 50%; transform: translateX(-50%);
    width: 5px; height: 5px; border-radius: 50%; background: var(--g300);
  }
  .jump a[data-state="done"]::after  { background: var(--olive); }
  .jump a[data-state="doing"]::after { background: var(--clay); }
  .jump a.here { background: var(--slate); color: var(--ivory); }
  .jump a.here::after { background: var(--ivory); }
  .jump a.flag { color: var(--clay-d); font-size: 15px; }
  .jump a.run { color: var(--olive); font-size: 12px; }
  .jump a.flag::after { display: none; }
  .jump .sep { width: 1.5px; height: 20px; background: var(--g200); margin: 0 2px; flex: none; }
  .jump button:focus-visible, .jump a:focus-visible { outline: 2px solid var(--clay); outline-offset: 2px; }
  .wrap { padding-bottom: 120px; }
  @media (min-width: 900px) { .wrap { padding-bottom: 140px; } }
  @media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } .chev { transition: none; } }
  html { scroll-behavior: smooth; scroll-padding-top: 16px; }

  /* ── prompts ── */
  .pbody { padding: 20px 22px 24px; }
  .pbody .sub { font-size: 13.5px; color: var(--g500); margin: 0 0 16px; max-width: 68ch; }
  .pbody .sub.foot { margin: 18px 0 0; padding-top: 14px; border-top: 1.5px solid var(--g200); }
  .pgroup { margin-bottom: 18px; }
  .pgroup h3 {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--g500); margin: 0 0 8px; font-weight: 400;
  }
  .pcard { border: 1.5px solid var(--g200); border-radius: 10px; background: var(--paper); margin-bottom: 6px; }
  .pcard.whole { border-color: var(--oat); background: #FBF6EC; }
  .pcard > summary {
    list-style: none; cursor: pointer; position: relative;
    display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
    padding: 11px 34px 11px 12px; min-height: 44px; border-radius: 9px;
  }
  .pcard > summary::-webkit-details-marker { display: none; }
  .pcard > summary:focus-visible { outline: 2px solid var(--clay); outline-offset: 2px; }
  .ptag {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase;
    padding: 3px 8px; border-radius: 5px; flex: none;
    color: var(--clay-d); background: #FBEDE6; border: 1px solid var(--oat);
  }
  .ptag.t { color: var(--g700); background: var(--g100); border-color: var(--g300); text-transform: none; letter-spacing: 0.04em; }
  .pname { font-size: 13.5px; color: var(--slate); min-width: 0; flex: 1 1 auto; }
  .pstate { font-family: var(--mono); font-size: 10px; color: var(--g500); flex: none; }
  .pcard .chev { top: 18px; }
  .pwrap { position: relative; padding: 0 12px 12px; }
  .pwrap pre {
    margin: 0; padding: 12px 14px; border-radius: 8px;
    border: 1.5px solid var(--g200); background: var(--g100); color: var(--g700);
    font-family: var(--mono); font-size: 11.5px; line-height: 1.65;
    white-space: pre; overflow-x: auto; -webkit-overflow-scrolling: touch;
  }
  button.copyp {
    position: absolute; top: -38px; right: 12px; z-index: 2;
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.06em;
    color: var(--g700); background: var(--paper); border: 1.5px solid var(--g300);
    border-radius: 8px; padding: 7px 12px; min-height: 32px; cursor: pointer;
  }
  button.copyp:hover { border-color: var(--g500); }
  button.copyp.ok { color: var(--olive); border-color: var(--olive); }

  @media (max-width: 560px) {
    /* wrap so it is readable on a phone. This is visual only - the copy button
       reads textContent, so the pasted prompt keeps its real line breaks and the
       VERIFY command stays on one line. */
    .pwrap pre { font-size: 10.5px; white-space: pre-wrap; overflow-wrap: anywhere; }
    button.copyp { top: -36px; right: 12px; padding: 6px 10px; }
    .pcard > summary { padding-right: 30px; }
    /* the whole bar must fit a 390px phone, or the decision flag falls off the end */
    .jump { padding-left: 6px; padding-right: 6px; }
    .jump .bar { gap: 1px; padding: 4px; }
    .jump a, .jump button { min-width: 31px; height: 36px; padding: 0 3px; font-size: 12.5px; }
    .jump .sep { display: none; }
    .jump a.flag { font-size: 14px; }
    .tasklist .tag { display: inline-block; margin: 2px 0 0; white-space: normal; }
  }
"""
if "task-tracker build additions" not in new:
    new = new.replace("  .hidden { display: none; }", EXTRA_CSS + "\n  .hidden { display: none; }")
LEGEND = ('<div class="legend"><span class="lgt">Done-when marks</span>'
          '<span class="lg"><span class="pip ok">D<i>&#10003;</i></span>passed</span>'
          '<span class="lg"><span class="pip bad">D<i>&#10007;</i></span>failing now</span>'
          '<span class="lg"><span class="pip">D<i>&middot;</i></span>not met</span>'
          '<span class="lg"><span class="pip unknown">D<i>?</i></span>needs a command to check</span>'
          '</div>')
if 'class="legend"' not in new:
    new = new.replace('<div class="timeline" id="timeline">', LEGEND + '\n    <div class="timeline" id="timeline">', 1)
# ---- bottom jump bar + the JS that drives it
jump_links = "".join(
    f'<a href="#p{p["n"]}" data-state="{"done" if p["status"]=="x" else ("doing" if p["status"]=="~" else "todo")}">{p["n"]}</a>'
    for p in phases
)
JUMP = f'''
<nav class="jump" aria-label="Jump to a phase">
  <div class="bar">
    <button id="jTop" title="Back to top" aria-label="Back to top">&uarr;</button>
    <span class="sep"></span>
    {jump_links}
    <span class="sep"></span>
    <a href="#prompts" class="flag run" title="Prompts you can run now" aria-label="Prompts you can run now">&#9654;</a>
    <a href="#you" class="flag" title="Needs your decision" aria-label="Needs your decision">&#9873;</a>
    <button id="jAll" title="Open or close every phase" aria-label="Open or close every phase">&#8597;</button>
  </div>
</nav>
'''

JUMP_JS = '''
<script>
(function () {
  var bar = document.querySelector('.jump');
  if (!bar) return;
  var links = Array.prototype.slice.call(bar.querySelectorAll('a[href^="#p"]'));

  // A filtered-out entry cannot be scrolled to. Reset the filter first.
  function showAll() {
    var all = document.querySelector('.chip[data-f="all"]');
    if (all && !all.classList.contains('on')) all.click();
  }
  bar.addEventListener('click', function (e) {
    var a = e.target.closest('a[href^="#"]');
    if (a) showAll();
  });

  document.getElementById('jTop').addEventListener('click', function () {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  var allOpen = false;
  document.getElementById('jAll').addEventListener('click', function () {
    allOpen = !allOpen;
    document.querySelectorAll('details.fold').forEach(function (d) { d.open = allOpen; });
  });

  // highlight the phase you are looking at
  var targets = links.map(function (a) { return document.querySelector(a.getAttribute('href')); }).filter(Boolean);
  if ('IntersectionObserver' in window && targets.length) {
    var seen = {};
    var io = new IntersectionObserver(function (rows) {
      rows.forEach(function (r) { seen[r.target.id] = r.isIntersecting; });
      var current = null;
      targets.forEach(function (t) { if (seen[t.id] && !current) current = t.id; });
      links.forEach(function (a) {
        a.classList.toggle('here', a.getAttribute('href') === '#' + current);
      });
    }, { rootMargin: '-10% 0px -70% 0px', threshold: 0 });
    targets.forEach(function (t) { io.observe(t); });
  }
})();
</script>
'''
# the prompt section goes last, just before the decision block
PA, PB = "<!-- BUILD:prompts -->", "<!-- /BUILD:prompts -->"
PROMPTS = PA + PROMPTS + PB
if PA in new and PB in new:
    new = new[: new.index(PA)] + PROMPTS + new[new.index(PB) + len(PB):]
elif '  <div class="next">' in new:
    new = new.replace('  <div class="next">', PROMPTS + '\n  <div class="next">', 1)
else:
    new = new.replace("  <footer>", PROMPTS + '\n  <footer>', 1)

PROMPT_JS = '''
<script>
(function () {
  document.querySelectorAll('.copyp').forEach(function (btn) {
    btn.addEventListener('click', function (e) {
      e.preventDefault(); e.stopPropagation();
      var pre = btn.parentElement.querySelector('pre');
      if (!pre) return;
      var text = pre.textContent;
      function done() {
        var old = btn.textContent;
        btn.textContent = 'copied'; btn.classList.add('ok');
        setTimeout(function () { btn.textContent = old; btn.classList.remove('ok'); }, 1600);
      }
      function fallback() {
        var ta = document.createElement('textarea');
        ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); } catch (err) {}
        document.body.removeChild(ta);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { fallback(); done(); });
      } else { fallback(); done(); }
    });
  });
})();
</script>
'''
if 'class="jump"' not in new:
    new = new.replace("</body>", JUMP + JUMP_JS + "</body>", 1)
if "copyp" not in new.split("<body>")[0] and ".copyp'" not in new:
    new = new.replace("</body>", PROMPT_JS + "</body>", 1)

# the owner-decision entry needs an id the flag button can reach
if 'data-t="human" id="you"' not in new:
    new = new.replace('<div class="entry t-human" data-t="human">',
                      '<div class="entry t-human" data-t="human" id="you">', 1)

# ---------------------------------------------------- stats and chip counts
# The tracker is the truth. A hand-typed stat row is how the page starts lying.
n_dev = new.count('data-t="dev"')
n_disc = new.count('data-t="disc"')
n_human = new.count('data-t="human"')
tasks_all = sum(len(p["tasks"]) for p in phases)
tasks_done = sum(1 for p in phases for t in p["tasks"] if t["mark"] in ("x", "-"))
cov_bad = sum(gates(p)[1]["bad"] for p in phases)

stats = (f'<div class="stat"><div class="n">{len(phases)}</div><div class="t">phases</div></div>'
         f'<div class="stat"><div class="n">{tasks_done}/{tasks_all}</div><div class="t">tasks done</div></div>'
         f'<div class="stat dev"><div class="n">{cov_bad}</div><div class="t">coverage rows failing</div></div>'
         f'<div class="stat you"><div class="n">{n_human}</div><div class="t">need your decision</div></div>')
new = re.sub(r'(<div class="doc-summary">).*?(</div>\s*\n\s*<div class="chips")',
             lambda m: m.group(1) + "\n      " + stats + "\n    </div>\n\n    <div class=\"chips\"",
             new, count=1, flags=re.S)

chips = (f'<button class="chip on" data-f="all">All <span class="ct">{len(phases)+n_dev+n_disc+n_human}</span></button>'
         f'<button class="chip" data-f="plan">Phases <span class="ct">{len(phases)}</span></button>'
         f'<button class="chip" data-f="dev">Deviations <span class="ct">{n_dev}</span></button>'
         f'<button class="chip" data-f="human">For you <span class="ct">{n_human}</span></button>')
new = re.sub(r'(<div class="chips"[^>]*>).*?(</div>)',
             lambda m: m.group(1) + "\n      " + chips + "\n    " + m.group(2),
             new, count=1, flags=re.S)

# ------------------------------------------------------- links back to source
# The page points at the two Markdown files; neither points back at the page.
# That direction is safe - a human clicks through, an agent never opens this.
rel = os.path.relpath(FOLDER.resolve(), page_p.parent.resolve())
new = re.sub(r'href="(?:\./|(?:\.\./)+[^"]*/)?((?:tracker|implementation)-[^"/]*\.md)"',
             lambda m: f'href="{rel}/{m.group(1)}"', new)

# ------------------------------------------------------------ stop banner
# The last and weakest layer. It cannot stop a full Read - by the time an agent
# sees this, the file is already in its context. It earns its place for partial
# reads, greps, and for whoever opens the file to hand-edit it. ADR 0052.
new = re.sub(r'<!-- =+\s*\n\s*STOP - AGENTS.*?-->\n?', '', new, flags=re.S)
BANNER = f"""<!-- ============================================================
     STOP - AGENTS DO NOT READ THIS FILE.

     This page is written for Hebert, and for him only. It is GENERATED
     from the two files that agents actually read:

       {rel}/{tracker_p.name}
       {rel}/{impl_p.name}

     It holds nothing those two do not already hold, and it costs roughly
     8,000 tokens of context to open. Reading it is pure waste.

     Need the state of the work?      read the tracker.
     Need the detail of a task?       read the implementation document.
     Need to change what is here?     edit that Markdown, then run
       .codex/skills/task-tracker/scripts/build-page.py <work-item-folder>

     Editing this file by hand is pointless - the next build overwrites it.
     Decision: ADR 0052.
     ============================================================ -->
"""
new = re.sub(r'<!-- generated-from:[^>]*-->\n?', '', new)
tsha = hashlib.sha1(tracker_p.read_bytes()).hexdigest()[:12]
isha = hashlib.sha1(impl_p.read_bytes()).hexdigest()[:12]
new = new.replace("<!doctype html>",
    BANNER + f"<!-- generated-from: tracker={tsha} implementation={isha} - DO NOT DELETE -->\n<!doctype html>", 1)

OUT.write_text(new)
print(f"built {OUT}")
print(f"  {len(phases)} phases, {tasks_done}/{tasks_all} tasks done, "
      f"{cov_bad} coverage rows failing, {n_human} needing a decision")
print(f"  {n_prompts} prompts for {len(prompt_blocks)} startable phase(s)"
      + (f"; {len(waiting)} phase(s) still waiting" if waiting else ""))
if cleanup_id == "?":
    print("  WARNING: no cleanup task found. The last phase must end with one.", file=sys.stderr)
