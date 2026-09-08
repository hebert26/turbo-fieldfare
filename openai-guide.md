# A simple guide to Codex

Checked against official OpenAI documentation on 7 September 2026.

**Start with the result you want, the relevant files, and what “done” means.**

This guide focuses on using Codex in the desktop app. Some linked OpenAI pages now use the ChatGPT name. Features depend on your app version, installed tools, and permissions.

## 1. Agent and subagent

An **agent** is the AI doing the work. In Codex, it can inspect files, use available tools, make changes, and check results. You use it by giving it a task in ordinary language.

> Explain how this app starts. Read the relevant code and give me five short points.

A **subagent** is another agent assigned one part of the work. The main agent coordinates the parts and combines the results.

| Use | Best for |
| --- | --- |
| One agent | A question, small edit, or focused bug fix. |
| Subagents | Separate investigations that can happen at the same time. |
| A separate task | Work you want to manage and continue independently. |

Ask for subagents like this:

```text
Use two subagents to investigate this problem.
One should inspect the relevant code.
One should check the official documentation.
Do not edit files. Wait for both.
Then explain the likely cause and next step.
```

More agents use more of your usage allowance. They can also conflict when editing the same files. Give each a clear responsibility. They keep the existing permission limits. See [OpenAI’s subagent guide](https://learn.chatgpt.com/docs/agent-configuration/subagents).

## 2. Give a clear request

A **prompt** is simply your request. Include the details that change the answer: the result, useful context, boundaries, and output format. See [OpenAI’s prompting guide](https://learn.chatgpt.com/docs/prompting).

Copy this and replace the brackets:

```text
I want: [the result].
Use: [files, screenshot, link, or project].
Keep unchanged: [important boundaries].
Done means: [something I can see or check].
Finish with a short explanation and a link to the result.
```

For a bug:

```text
The app closes when I press Send with an empty message.
Find the cause and fix it.
Done means the app stays open and explains what I need to enter.
Keep the change focused and run the relevant checks.
```

Attach the error or screenshot when possible. You can correct the direction while Codex works:

> Focus only on the Send button. Keep your updates short.

## 3. Useful capabilities

These are examples to try when the matching tools are available. Codex should tell you when a required connection or tool is missing. See [OpenAI’s capabilities overview](https://learn.chatgpt.com/docs/features).

| What you need | What to ask |
| --- | --- |
| Understand code | “Explain what happens after I press Send.” |
| Change an app | “Add this button and check that it works.” |
| Research | “Check the official documentation and link your sources.” |
| Create a guide | “Write a beginner guide and save it as a Markdown file.” |
| Work with files | “Turn this spreadsheet into a short report.” |
| Explain visually | “Create a simple labelled image explaining this process.” |
| Inspect a screen | “Use this screenshot to identify confusing controls.” |
| Check a website | “Open this page and test the sign-up flow.” |

For code review, name the changes to inspect and ask for evidence. The review pane shows changed lines, also called a **diff**. See [OpenAI’s code review guide](https://learn.chatgpt.com/docs/code-review).

> Review the changes from your last turn for bugs. Explain each finding with its file and line. Do not edit yet.

## 4. Save instructions you repeat

**AGENTS.md** stores project instructions that Codex reads when starting work. Use it for writing preferences, project rules, and required commands. See [OpenAI’s AGENTS.md guide](https://learn.chatgpt.com/docs/agent-configuration/agents-md).

This project already has [AGENTS.md](/Users/dev-machine/dev/turbo-fieldfare-personal/AGENTS.md). It includes short-response preferences and requires only one model process at a time.

**A skill** stores a repeatable way of doing a particular job. **A plugin** is an installable bundle that can add skills or connections to other services. **MCP** is the connection standard behind many of those tools. See [OpenAI’s skills and plugins guide](https://learn.chatgpt.com/docs/skills-and-plugins).

Use an installed skill by name:

> Use the openai-docs skill to check this answer.

Or ask for a reusable workflow:

> Use $skill-creator to create a skill for my weekly project report. Include completed work, blockers, and next steps.

## 5. Keep work organised

A **project** groups related work and files. A **task**, also called a chat, holds one conversation. Continue the same task for follow-ups. Start a new one for a different outcome. Save lasting decisions in files so future tasks can read them. See [OpenAI’s projects guide](https://learn.chatgpt.com/docs/projects).

You may also see these choices:

| Choice | Where changes happen |
| --- | --- |
| Local | Directly in your project folder. |
| Worktree | In a separate Git checkout on your computer. |
| Cloud | In a configured remote environment. |

A worktree helps separate simultaneous code changes. A cloud environment needs its own setup. See [OpenAI’s environment guide](https://learn.chatgpt.com/docs/environments/modes).

For this TurboFieldfare project, follow its existing instructions before changing environments or running the model.

## 6. Planning, goals, and scheduled work

Use **Plan mode** when you want to settle the approach before making changes. Use **Goal mode** for work with several steps and a clear finish. Where supported, enter `/plan` or `/goal`. Goals keep the existing permission limits. See [OpenAI’s long-running work guide](https://learn.chatgpt.com/docs/long-running-work).

Planning example:

> Plan how to add image attachments. Name the screens and files involved. Wait before implementing.

Goal example:

> Create a goal to fix the empty-message crash. Finish when the app stays open and the relevant checks pass.

A **scheduled task** runs later or repeats. State the timing and when you want a notification. Local scheduled work needs the computer on and the app running. Check that the schedule was actually created. See [OpenAI’s scheduled tasks guide](https://learn.chatgpt.com/docs/automations).

> Every Monday at 9 am, Europe/London time, review this project’s documentation for broken local file links. Notify me only if you find any. Do not edit files.

## 7. My recommended everyday habit

Before accepting a result, ask:

> What changed, what did you check, and what remains unfinished? Keep it short.

For work you want to review before it leaves your computer, add:

> Prepare everything locally. Ask before publishing, pushing code, or sending messages.

**Choose one concrete result. Give Codex the relevant context. Ask it to finish and show the evidence.**
