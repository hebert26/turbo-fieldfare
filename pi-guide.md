# Pi in this repo

Pi is the terminal coding agent. Project config lives in `.pi/`. Start it from this folder so it loads that config.

## Start Pi in tmux

Subagents need tmux. Start Pi inside a tmux session, not in a plain terminal.

Install tmux once:

```bash
brew install tmux
```

From this repo, this is the command:

```bash
cd /Users/dev-machine/dev/turbo-fieldfare-personal
tmux new -A -s pi 'pi'
```

What that line means:

| Piece | Meaning |
|---|---|
| `tmux new` | Start a tmux session |
| `-A` | If a session with this name already exists, attach to it instead of making a second one |
| `-s pi` | Session name. Use a different name if you also run Pi in another repo, for example `-s tff` |
| `'pi'` | The program to run inside the session (the Pi CLI) |

`-A` makes this command reuse the session named `pi`. It does not start a fresh Pi when that session already exists.

### Start a fresh separate session

Choose a name that does not already exist. This starts another Pi in its own tmux session:

```bash
tmux new -s pi-fresh 'pi'
```

Replace `pi-fresh` with a name you will remember. Reconnect to it with `tmux attach -t '=pi-fresh'`.

For a new name every time, let macOS generate one:

```bash
name="pi-$(uuidgen | tr '[:upper:]' '[:lower:]')"
echo "Reconnect with: tmux attach -t $name"
tmux new -s "$name" 'pi'
```

### Use hyphens in session names

Use letters, numbers, and hyphens, such as `qwen3-6-35b-a3b`. Avoid dots and colons: tmux uses `:` to separate a session from a window and `.` to identify a pane. A dotted name such as `Qwen3.6-35B-A3B` can be interpreted as a pane target and cause `can't find pane: 6-35B-A3B`.

From a terminal outside tmux, start or reconnect with:

```bash
cd /Users/dev-machine/dev/turbo-fieldfare-personal
tmux new -A -s qwen3-6-35b-a3b 'pi'
```

The session `qwen3-6-35b-a3b` already exists. To attach directly:

```bash
tmux attach -t '=qwen3-6-35b-a3b'
```

The `=` tells tmux to match the exact session name rather than a prefix or pattern. Quotes keep the target literal in the shell. The session name is only a label; it does not select Pi's model.

If you are already inside tmux, switch sessions instead of nesting another attachment:

```bash
tmux switch-client -t '=qwen3-6-35b-a3b'
```

### Replace the `pi` session

This stops Pi and closes every window in the old `pi` session. Then it creates a new one:

```bash
tmux kill-session -t pi
tmux new -s pi 'pi'
```

### Add a window to the current session

While attached to `pi`, press `Ctrl+B` then `c`. tmux opens a new window. Run Pi from this repo in that window:

```bash
cd /Users/dev-machine/dev/turbo-fieldfare-personal
pi
```

Come back later:

```bash
tmux attach -t pi
```

Leave without stopping Pi: `Ctrl+B` then `D`.

If you are already inside that tmux session:

```bash
cd /Users/dev-machine/dev/turbo-fieldfare-personal
pi
```

Other tmux keys (prefix is `Ctrl+B`):

| Keys | What it does |
|---|---|
| `Ctrl+B` then `D` | Detach. Pi keeps running. |
| `tmux attach -t pi` | Come back to the session named `pi`. |
| `Ctrl+B` then `→` / `←` | Move between panes (main Pi vs a subagent). |
| `Ctrl+B` then `X` | Close the current pane. |
| `Ctrl+B` then `[` | Scroll the pane. `Q` to leave scroll. |

First run: trust this project when Pi asks. That loads `.pi/settings.json` and the local extensions.

After you edit an extension, run `/reload` or restart Pi.

## Everyday Pi keys

| Keys / input | What it does |
|---|---|
| Type a request, Enter | Send it |
| `@` | Attach a project file |
| `/` | Commands |
| `!command` | Run shell and send the output to the model |
| `!!command` | Run shell, keep output out of chat |
| `Ctrl+L` | Pick a model |
| `Shift+Tab` | Cycle thinking level |
| `Escape` | Stop the current run |
| `Ctrl+C` twice | Quit |
| `/new` | New session |
| `/resume` | Old session |
| `/reload` | Reload extensions and context |

`/hotkeys` shows the rest.

## What this project loads

`.pi/settings.json` turns **off** project skills. It turns **on** only these extensions:

| Extension | What you get |
|---|---|
| `custom-header.ts` | Custom startup header. `/builtin-header` restores the stock one. |
| `voice.ts` | Spoken replies. `/voice` on, `/voice off`, `/voice doctor`. |
| `ask-user-question.ts` | Model can pause and ask you one question (`ask_user_question`). |
| `interactive-subagents/` | Spawn workers in extra tmux panes. |

Context on every turn:

- `.pi/APPEND_SYSTEM.md` — communication rules and aliases
- `AGENTS.md` — this repo’s work rules

Other `.ts` files under `.pi/extensions/` stay on disk but stay off until you add a `+extensions/name.ts` line in `.pi/settings.json`.

## Voice

```
/voice
/voice off
/voice doctor
```

`/voice` needs `python3` and `edge-tts`. The speak script is `.pi/skills/voice/scripts/speak.py`.

Say `eli` on its own if you want a short, simple answer. `speak` means: turn voice on and explain that way.

## Aliases (type them alone)

These come from `.pi/APPEND_SYSTEM.md`.

| You type | Pi should |
|---|---|
| `eli` | Explain simply and short |
| `foc` | Keep only the main point |
| `scr` | Compress and repeat |
| `ref` | Use numbered reference codes |
| `speak` | Use `/voice` and `eli` |
| `why` | Explain only. Do not take action |

If the word sits inside a longer sentence, it is not an alias.

## Ask you a question

The model can call `ask_user_question` when a choice matters. You get a picker in the TUI. Arrow keys, Enter, Esc. Options can include Other.

## Subagents (needs tmux)

Only works when the parent `pi` is inside tmux.

Spawn from chat, or with a command:

```
/subagent scout Find where Gemma decode is launched
/subagent worker Add a flag to skip the model download
```

The model can also call tools:

- `subagent` — start a worker in a new pane
- `subagent_message` — talk to one by name
- `subagents_list` — list known agents

A live widget above the input shows running workers. When one finishes, its result comes back into the main session.

### Bundled agents

These live in `.pi/extensions/interactive-subagents/agents/`:

| Agent | Role |
|---|---|
| `scout` | Read-only search |
| `researcher` | Web research (needs web tools) |
| `worker` | Edit code. May spawn scout and researcher |

Edit those markdown files to change models, tools, or instructions.

### Project agents

Markdown files in `.pi/agents/` also count as spawnable agents. A project file with the same `name` as a bundled agent wins.

Example:

```markdown
---
name: my-helper
description: Short job this agent is for.
tools: read,grep,find,ls
model: openai-codex/gpt-5.5
thinking: low
auto-exit: true
system-prompt: append
---

What this agent should do.
```

Then: `/subagent my-helper Your task here`.

## Change what loads

Edit `.pi/settings.json`.

Turn an extra local extension on:

```json
"+extensions/footer.ts"
```

Turn one off by deleting its `+extensions/...` line. Empty `"extensions": []` does **not** turn auto-load off.

After a settings or extension edit: `/reload`.

## Edit the local subagent code

The source is `.pi/extensions/interactive-subagents/`.

| File | What to change |
|---|---|
| `index.ts` | Tools and spawn logic |
| `tmux.ts` | Pane layout |
| `status.ts`, `activity.ts` | Widget |
| `agents/*.md` | Bundled roles |
| `config.json` | Status widget on/off |

This is a local copy, not an installed package. Change the files here.
