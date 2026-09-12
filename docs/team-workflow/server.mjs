import { createServer } from "node:http";
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, readSync, readdirSync, statSync, watch } from "node:fs";
import { homedir } from "node:os";
import { join, basename } from "node:path";
import { fileURLToPath } from "node:url";

const HOST = "127.0.0.1";
const PORT = Number(process.env.PORT || 47831);
const TEAM_CHAT_DIR = process.env.TEAM_CHAT_DIR || join(homedir(), ".pi", "agent", "team-chat");
const PROJECT_SESSION_ROOT = join(homedir(), ".pi", "agent", "sessions", "--Users-dev-machine-dev-turbo-fieldfare-personal--");
const REGISTRY_PATH = process.env.PI_SUBAGENT_REGISTRY || join(PROJECT_SESSION_ROOT, "artifacts", "01a0911c-cbe7-73f7-a5d2-6bde960a889a", "subagent-registry.json");
const ACTIVITY_TAIL_BYTES = 512 * 1024;
const ACTIVITY_MAX_EVENTS = 160;
const RUNNING_WINDOW_MS = 30 * 1000;
const HTML_PATH = join(fileURLToPath(new URL(".", import.meta.url)), "index.html");
const WORKFLOW_TYPES = new Set([
  "roster_set", "join", "split_proposed", "split_reviewed", "plan_submitted", "plan_reviewed",
  "initial_review_submitted", "initial_review_compared", "escalation_requested", "escalation_verdict",
  "verification_submitted", "verification_reviewed", "finish", "claim", "release"
]);
const clients = new Set();
let lastSignature = "";

function finite(value) { return typeof value === "number" && Number.isFinite(value); }
function text(value, fallback = "") { return typeof value === "string" ? value : fallback; }
function eventIndex(event, fallback) { return finite(event.index) ? event.index : fallback; }
function short(value, limit = 260) {
  const compact = text(value).replace(/\s+/g, " ").trim();
  return compact.length <= limit ? compact : `${compact.slice(0, limit - 1)}…`;
}
function owners(roster, duty) { return roster.filter(member => member.duties.includes(duty)).map(member => member.name); }
function latest(events, type, predicate = () => true) {
  return [...events].reverse().find(event => event.type === type && predicate(event));
}
function accepted(events, submittedType, reviewedType, owner, decision) {
  const proposal = latest(events, submittedType);
  if (!proposal || !owner) return false;
  return Boolean(latest(events, reviewedType, event => event.sender === owner && event.proposal_index === eventIndex(proposal, -1) && event.decision === decision));
}
function sameNumbers(a = [], b = []) {
  const left = Array.isArray(a) ? a.filter(finite).sort((x, y) => x - y) : [];
  const right = Array.isArray(b) ? b.filter(finite).sort((x, y) => x - y) : [];
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function parseJournal(path, teamId) {
  const warnings = [];
  let source = "";
  try { source = readFileSync(path, "utf8"); }
  catch (error) { return { teamId, events: [], warnings: [`Unable to read journal: ${error.message}`], mtime: 0 }; }
  const hasFinalNewline = /(?:\r?\n)$/.test(source);
  const rawLines = source.split(/\r?\n/);
  if (hasFinalNewline) rawLines.pop();
  const events = [];
  rawLines.forEach((line, lineIndex) => {
    if (!line.trim()) return;
    if (!hasFinalNewline && lineIndex === rawLines.length - 1) {
      warnings.push(`Ignored transient partial final line ${lineIndex + 1}; waiting for a newline.`);
      return;
    }
    try {
      const parsed = JSON.parse(line);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed) || typeof parsed.type !== "string") {
        warnings.push(`Ignored invalid record on line ${lineIndex + 1}.`);
        return;
      }
      if (parsed.team_id && parsed.team_id !== teamId) {
        warnings.push(`Ignored record for another team on line ${lineIndex + 1}.`);
        return;
      }
      events.push({ ...parsed, index: eventIndex(parsed, lineIndex), team_id: teamId });
    } catch {
      warnings.push(`Ignored malformed record on line ${lineIndex + 1}.`);
    }
  });
  let mtime = 0;
  try { mtime = statSync(path).mtimeMs; } catch {}
  return { teamId, events, warnings, mtime };
}

function discoverJournals() {
  if (!existsSync(TEAM_CHAT_DIR)) return [];
  let names = [];
  try { names = readdirSync(TEAM_CHAT_DIR).filter(name => /^[A-Za-z0-9._-]{1,120}\.jsonl$/.test(name)); } catch { return []; }
  return names.map(name => parseJournal(join(TEAM_CHAT_DIR, name), basename(name, ".jsonl")));
}

function derive(events, config) {
  let roster = [];
  const joined = new Set();
  let workflow = [];
  for (const event of events) {
    if (event.type === "roster_set") {
      roster = Array.isArray(event.members) ? event.members.filter(member => member && typeof member.name === "string").map(member => ({
        name: member.name, duties: Array.isArray(member.duties) ? member.duties.filter(value => typeof value === "string") : [], work: text(member.work)
      })) : [];
      for (const name of [...joined]) if (!roster.some(member => member.name === name)) joined.delete(name);
      if (event.reset_from === "dividing") workflow = [];
      else if (event.reset_from === "plan_challenge") workflow = workflow.filter(item => ["split_proposed", "split_reviewed", "plan_submitted"].includes(item.type));
      else if (event.reset_from === "verification_challenge") workflow = workflow.filter(item => item.type !== "verification_reviewed" && item.type !== "finish");
    } else if (event.type === "join" && roster.some(member => member.name === event.sender)) {
      joined.add(event.sender);
    } else if (config && roster.some(member => member.name === event.sender)) {
      workflow.push(event);
    }
  }
  const members = roster.filter(member => joined.has(member.name));
  const implementation = owners(roster, "implementation")[0];
  const planReviewer = owners(roster, "plan_review")[0];
  const initialReviewers = [owners(roster, "initial_review_luna")[0], owners(roster, "initial_review_terra")[0]].filter(Boolean);
  const verifiers = owners(roster, "verification");
  const splitAccepted = accepted(workflow, "split_proposed", "split_reviewed", planReviewer, "accept");
  const planApproved = accepted(workflow, "plan_submitted", "plan_reviewed", planReviewer, "approve");
  const plan = latest(workflow, "plan_submitted");
  const initialReviews = initialReviewers.map(reviewer => latest(workflow, "initial_review_submitted", event => event.sender === reviewer && event.proposal_index === eventIndex(plan, -1)));
  const initialIndexes = initialReviews.map(event => event?.index).filter(finite).sort((a, b) => a - b);
  const initialComplete = initialReviews.length === 2 && initialIndexes.length === 2;
  const comparisons = initialComplete ? initialReviewers.map(reviewer => {
    const peer = initialReviews.find(event => event?.sender !== reviewer);
    return peer ? latest(workflow, "initial_review_compared", event => event.sender === reviewer && event.proposal_index === eventIndex(peer, -1)) : undefined;
  }) : [];
  const comparisonsComplete = comparisons.length === 2 && comparisons.every(Boolean);
  const disagreement = comparisons.some(event => event?.decision === "challenge");
  const escalationRequest = latest(workflow, "escalation_requested", event => sameNumbers(event.initial_review_indexes, initialIndexes));
  const escalationVerdict = escalationRequest ? latest(workflow, "escalation_verdict", event => event.proposal_index === eventIndex(escalationRequest, -1)) : undefined;
  const verification = latest(workflow, "verification_submitted");
  const verificationReviews = verification ? workflow.filter(event => event.type === "verification_reviewed" && event.proposal_index === eventIndex(verification, -1)) : [];
  const verificationChallenged = verificationReviews.some(event => event.decision === "challenge" && verifiers.includes(event.sender));
  const verificationApproved = Boolean(verification) && !verificationChallenged && verifiers.every(verifier => verificationReviews.some(event => event.sender === verifier && event.decision === "approve"));
  let stage = "forming";
  const required = roster.filter(member => !member.duties.includes("escalation"));
  if (config && required.every(member => joined.has(member.name)) && implementation && planReviewer && initialReviewers.length === 2 && verifiers.length > 0) {
    if (!splitAccepted) stage = "dividing";
    else if (!planApproved) stage = "plan_challenge";
    else if (!initialComplete) stage = "initial_review";
    else if (escalationVerdict?.verdict === "block") stage = "initial_review_resolution";
    else if (!comparisonsComplete) stage = "initial_review_peer_challenge";
    else if ((escalationRequest && !escalationVerdict) || (disagreement && !escalationVerdict)) stage = "escalation";
    else if (!verification || verificationChallenged) stage = "implementation";
    else if (!verificationApproved) stage = "verification_challenge";
    else stage = "completed";
  }
  return { stage, goal: text(config?.goal, "No goal recorded"), mode: text(config?.mode, "implementation"), revision: config?.revision ?? 0, roster, members, workflow, splitAccepted, planApproved, verificationApproved, initialReviews, comparisons, escalationRequest, escalationVerdict, verification, verificationReviews };
}

function terminalEvidence(events, state) {
  const records = events.filter(event => event.type === "message" && event.status === "member-terminal" && typeof event.member_name === "string");
  const members = state.members.map(member => {
    const record = [...records].reverse().find(event => event.member_name === member.name);
    const status = record?.terminal_status || "missing";
    return { name: member.name, status, message: short(record?.message, 220), index: record?.index ?? null };
  });
  const finishes = events.filter(event => event.type === "finish").map(event => ({ status: text(event.status, "unknown"), terminalStatus: text(event.terminal_status), message: short(event.message, 220), index: event.index ?? null }));
  const cleanMembers = members.length > 0 && members.every(member => member.status === "complete");
  const cleanFinishes = finishes.every(finish => finish.status === "complete");
  return { available: cleanMembers && cleanFinishes, members, finishes };
}

function conversationText(event) {
  if (event.type === "message") return typeof event.message === "string" ? event.message : "";
  const parts = [];
  if (event.type === "split_proposed") parts.push(...(Array.isArray(event.assignments) ? event.assignments.map(item => `${item.name}: ${item.work}`) : []));
  if (event.type === "plan_submitted") parts.push(`Plan:\n${text(event.plan)}\n\nEvidence:\n${text(event.evidence)}\n\nRisk:\n${text(event.risk)}\n\nCheck:\n${text(event.check)}`);
  if (event.type === "initial_review_submitted") parts.push(`Findings:\n${text(event.findings)}\n\nEvidence:\n${text(event.evidence)}\n\nRisk:\n${text(event.risk)}`);
  if (event.type === "verification_submitted") parts.push(`Outcome:\n${text(event.outcome)}\n\nEvidence:\n${text(event.evidence)}\n\nChecks:\n${text(event.checks)}\n\nRemaining risk:\n${text(event.remaining_risk)}`);
  if (event.type === "split_reviewed" || event.type === "plan_reviewed" || event.type === "initial_review_compared") parts.push(`${text(event.decision)}${event.message ? ` — ${event.message}` : ""}`);
  if (event.type === "escalation_requested") parts.push(`Requested Terra high escalation: ${text(event.reason)}`);
  if (event.type === "escalation_verdict") parts.push(`${text(event.verdict)} — ${text(event.evidence)}`);
  if (event.type === "verification_reviewed") parts.push(`${text(event.decision)}${event.message ? ` — ${event.message}` : ""}`);
  return parts.join("\n");
}
function conversationEvents(selected, state) {
  const all = [];
  const add = event => {
    const message = conversationText(event);
    if (!message || !finite(event.ts)) return;
    if (event.type === "message" && (event.status === "member-terminal" || event.status === "final-notified")) return;
    if (event.type === "message" && event.sender === "main") return;
    all.push({ index: event.index ?? null, ts: event.ts, sender: text(event.sender, "unknown"), recipients: Array.isArray(event.recipients) && event.recipients.length ? event.recipients : ["Team"], type: event.type, message });
  };
  for (const event of selected.events) if (event.type === "message") add(event);
  for (const event of state.workflow) if (event.type !== "message" && ["split_proposed", "split_reviewed", "plan_submitted", "plan_reviewed", "initial_review_submitted", "initial_review_compared", "escalation_requested", "escalation_verdict", "verification_submitted", "verification_reviewed"].includes(event.type)) add(event);
  return [...new Map(all.map(event => [event.index ?? `${event.type}:${event.ts}:${event.sender}`, event])).values()].sort((a, b) => (a.ts - b.ts) || ((a.index ?? 0) - (b.index ?? 0)));
}
function displayEvent(event) {
  const labels = {
    roster_set: "Roster configured", join: "Member joined", split_proposed: "Assignments proposed", split_reviewed: "Split decision",
    plan_submitted: "Plan submitted", plan_reviewed: "Plan decision", initial_review_submitted: "Initial review", initial_review_compared: "Peer comparison",
    escalation_requested: "Escalation requested", escalation_verdict: "Escalation verdict", verification_submitted: "Verification submitted",
    verification_reviewed: "Verification decision", finish: "Finish record", claim: "File claimed", release: "File released"
  };
  const details = event.type === "roster_set" ? text(event.goal) : event.type === "split_reviewed" || event.type === "plan_reviewed" || event.type === "initial_review_compared" ? `${text(event.decision)} — ${short(event.message)}` : event.type === "verification_reviewed" ? `${text(event.decision)} — ${short(event.message)}` : event.type === "escalation_verdict" ? `${text(event.verdict)} — ${short(event.evidence)}` : event.type === "escalation_requested" ? short(event.reason) : event.type === "verification_submitted" ? short(event.outcome) : event.type === "finish" ? `${text(event.status, "unknown")} ${short(event.message)}` : event.type === "claim" || event.type === "release" ? text(event.file_path) : event.type === "join" ? "" : event.type === "initial_review_submitted" ? short(event.findings) : event.type === "plan_submitted" ? short(event.plan) : "";
  return { index: event.index ?? null, ts: finite(event.ts) ? event.ts : null, type: event.type, label: labels[event.type] || event.type, sender: text(event.sender, "unknown"), decision: text(event.decision || event.verdict), details };
}

function readTail(path, limit = ACTIVITY_TAIL_BYTES) {
  try {
    const info = statSync(path);
    const size = Math.min(info.size, limit);
    const fd = openSync(path, "r");
    const buffer = Buffer.alloc(size);
    readSync(fd, buffer, 0, size, Math.max(0, info.size - size));
    closeSync(fd);
    return { text: buffer.toString("utf8"), mtime: info.mtimeMs, size: info.size, truncated: info.size > limit };
  } catch { return { text: "", mtime: 0, size: 0, truncated: false }; }
}
function readHead(path, limit = 64 * 1024) {
  try {
    const info = statSync(path);
    const size = Math.min(info.size, limit);
    const fd = openSync(path, "r");
    const buffer = Buffer.alloc(size);
    readSync(fd, buffer, 0, size, 0);
    closeSync(fd);
    return { text: buffer.toString("utf8"), size: info.size, mtime: info.mtimeMs };
  } catch { return { text: "", size: 0, mtime: 0 }; }
}
function epoch(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value < 1e12 ? value * 1000 : value;
  if (typeof value === "string") { const parsed = Date.parse(value); return Number.isFinite(parsed) ? parsed : 0; }
  return 0;
}
function safeAction(name) {
  if (name === "bash") return "run validation";
  if (name === "read") return "inspect source";
  if (name === "write") return "create artifact";
  if (name === "edit") return "update file";
  if (name === "grep") return "search source";
  if (name === "find" || name === "ls") return "inspect files";
  if (name === "open") return "open browser";
  if (name.startsWith("team_")) {
    const labels = { team_read_messages: "read workflow", team_status: "inspect team", team_send_message: "coordinate team", team_claim_file: "claim file", team_propose_split: "propose assignments", team_submit_plan: "submit plan", team_submit_verification: "submit verification" };
    return labels[name] || name.slice(5).replaceAll("_", " ");
  }
  return "run tool";
}
function readRegistry() {
  const warnings = [];
  let registry;
  try { registry = JSON.parse(readFileSync(REGISTRY_PATH, "utf8")); }
  catch { return { entries: {}, warnings: ["Activity registry unavailable; activity mapping is unavailable."] }; }
  if (!registry || typeof registry !== "object" || Array.isArray(registry)) return { entries: {}, warnings: ["Activity registry is invalid; activity mapping is unavailable."] };
  const entries = {};
  for (const [name, value] of Object.entries(registry)) {
    if (value && typeof value === "object" && typeof value.sessionFile === "string" && value.sessionFile) entries[name] = value.sessionFile;
  }
  return { entries, warnings };
}
const teamEvidenceCache = new Map();
function hasTeamEvidence(path, teamId, mtime, size) {
  const key = `${path}:${teamId}`;
  const cached = teamEvidenceCache.get(key);
  if (cached && cached.mtime === mtime && cached.size === size) return cached.found;
  let found = false;
  try {
    const head = readHead(path, 512 * 1024);
    const tail = readTail(path, 512 * 1024);
    found = `${head.text}\n${tail.text}`.includes(teamId);
  } catch {}
  teamEvidenceCache.set(key, { mtime, size, found });
  return found;
}
function parseActivitySession(path, joinTs, now, teamId) {
  const tail = readTail(path);
  if (!tail.text) return { mapped: false, events: [], mtime: tail.mtime, warnings: ["Mapped session could not be read."] };
  if (!hasTeamEvidence(path, teamId, tail.mtime, tail.size)) return { mapped: false, events: [], mtime: tail.mtime, warnings: ["Mapped session has no exact current-team evidence."] };
  const lines = tail.text.split(/\r?\n/);
  if (tail.truncated && lines.length) lines.shift();
  const calls = new Map();
  const warnings = [];
  for (const line of lines) {
    if (!line.trim()) continue;
    let outer;
    try { outer = JSON.parse(line); } catch { warnings.push("Ignored malformed session record."); continue; }
    const timestamp = epoch(outer.timestamp ?? outer.ts);
    const message = outer.message;
    if (!message || typeof message !== "object") continue;
    if (message.role === "assistant" && Array.isArray(message.content)) {
      for (const item of message.content) {
        if (item?.type !== "toolCall" || typeof item.id !== "string" || typeof item.name !== "string" || timestamp < joinTs) continue;
        calls.set(item.id, { id: item.id, tool: item.name, action: safeAction(item.name), ts: timestamp, result: null });
      }
    }
    if (message.role === "toolResult" && typeof message.toolCallId === "string") {
      const call = calls.get(message.toolCallId);
      if (call && timestamp >= joinTs) call.result = { ts: timestamp, error: message.isError === true };
    }
  }
  const events = [...calls.values()].map(call => {
    const result = call.result;
    let state = result ? (result.error ? "failed" : "completed") : "unknown";
    if (!result && call.ts > 0 && tail.mtime > 0 && now - call.ts <= RUNNING_WINDOW_MS && now - tail.mtime <= RUNNING_WINDOW_MS) state = "running";
    return { ...call, state, ageSeconds: call.ts > 0 ? Math.max(0, Math.round((now - call.ts) / 1000)) : null };
  }).sort((a, b) => a.ts - b.ts);
  return { mapped: true, events, mtime: tail.mtime, warnings };
}
function activitySnapshot(selected, state) {
  const now = Date.now();
  const registry = readRegistry();
  const warnings = [...registry.warnings];
  const joins = new Map(selected.events.filter(event => event.type === "join").map(event => [event.sender, epoch(event.ts)]));
  const lanes = [];
  const feed = [];
  for (const member of state.roster) {
    const sessionPath = registry.entries[member.name];
    const joinTs = joins.get(member.name) || 0;
    if (!sessionPath || !joinTs) {
      lanes.push({ name: member.name, joined: state.members.some(active => active.name === member.name), duties: member.duties, assignment: member.work, mapped: false, status: "activity unavailable", currentAction: "No exact session mapping", ageSeconds: null });
      continue;
    }
    const activity = parseActivitySession(sessionPath, joinTs, now, selected.teamId);
    warnings.push(...activity.warnings.map(warning => `${member.name}: ${warning}`));
    const latestEvent = activity.events.at(-1);
    let status = "no active tool observed";
    let action = "No active tool observed";
    if (!activity.mapped) status = "activity unavailable", action = "No exact current-team evidence";
    else if (latestEvent?.state === "running") status = "running", action = latestEvent.action;
    else if (latestEvent?.state === "unknown") status = "unknown (stale/unresolved)", action = latestEvent.action;
    else if (latestEvent?.state === "failed") status = "failed", action = latestEvent.action;
    else if (latestEvent) action = latestEvent.action;
    const lane = { name: member.name, joined: state.members.some(active => active.name === member.name), duties: member.duties, assignment: member.work, mapped: activity.mapped, status, currentAction: action, ageSeconds: latestEvent?.ageSeconds ?? null };
    lanes.push(lane);
    activity.events.slice(-ACTIVITY_MAX_EVENTS).forEach(event => feed.push({ tool: event.tool, action: event.action, ts: event.ts, state: event.state, ageSeconds: event.ageSeconds, sender: member.name }));
  }
  feed.sort((a, b) => b.ts - a.ts);
  return { lanes, feed: feed.slice(0, 80), warnings: [...new Set(warnings)] };
}
function snapshot() {
  const journals = discoverJournals();
  const candidates = journals.flatMap(journal => journal.events.filter(event => event.type === "roster_set" && finite(event.ts)).map(event => ({ journal, event })));
  if (candidates.length === 0) return { available: false, generatedAt: Date.now(), source: null, warnings: journals.flatMap(journal => journal.warnings).concat(journals.length ? ["No valid roster configuration found."] : ["No team journals found."]), state: null };
  candidates.sort((a, b) => (b.event.ts - a.event.ts) || (b.journal.mtime - a.journal.mtime));
  const selected = candidates[0].journal;
  const config = [...selected.events].reverse().find(event => event.type === "roster_set" && finite(event.ts));
  const state = derive(selected.events, config);
  const terminal = terminalEvidence(selected.events, state);
  const activity = activitySnapshot(selected, state);
  const conversation = conversationEvents(selected, state);
  const claims = {};
  for (const event of state.workflow) {
    if (!event.file_path) continue;
    if (event.type === "claim") claims[event.file_path] = event.sender;
    if (event.type === "release") delete claims[event.file_path];
  }
  return {
    available: true, generatedAt: Date.now(), source: selected.teamId,
    warnings: [...selected.warnings, ...activity.warnings], state: { stage: state.stage, goal: state.goal, mode: state.mode, revision: state.revision, roster: state.roster.map(member => ({ ...member, joined: state.members.some(active => active.name === member.name) })), splitAccepted: state.splitAccepted, planApproved: state.planApproved, verificationApproved: state.verificationApproved, workflow: state.workflow.filter(event => WORKFLOW_TYPES.has(event.type)).map(displayEvent), terminal, claims, activity, conversation, verification: state.verification ? { outcome: short(state.verification.outcome, 500), evidence: short(state.verification.evidence, 700), checks: short(state.verification.checks, 700), remainingRisk: short(state.verification.remaining_risk, 500), reviews: state.verificationReviews.map(event => ({ sender: event.sender, decision: event.decision, message: short(event.message, 400) })) } : null, escalation: state.escalationRequest || state.escalationVerdict ? { requested: state.escalationRequest ? { sender: state.escalationRequest.sender, reason: short(state.escalationRequest.reason, 500) } : null, verdict: state.escalationVerdict ? { sender: state.escalationVerdict.sender, verdict: state.escalationVerdict.verdict, evidence: short(state.escalationVerdict.evidence, 500) } : null } : null }
  };
}

function signature() {
  const parts = [];
  try { const info = statSync(REGISTRY_PATH); parts.push(`registry:${info.size}:${info.mtimeMs}`); } catch { parts.push("registry:missing"); }
  try { parts.push(...Object.values(readRegistry().entries).map(path => { try { const info = statSync(path); return `session:${info.size}:${info.mtimeMs}`; } catch { return "session:missing"; } })); } catch {}
  try { parts.push(...readdirSync(TEAM_CHAT_DIR).filter(name => name.endsWith(".jsonl")).map(name => { const info = statSync(join(TEAM_CHAT_DIR, name)); return `${name}:${info.size}:${info.mtimeMs}`; })); } catch { parts.push("journals:missing"); }
  return parts.join("|");
}
function notify() {
  const next = signature();
  if (next === lastSignature) return;
  lastSignature = next;
  const payload = `data: ${JSON.stringify(snapshot())}\n\n`;
  for (const response of clients) response.write(payload);
}
function json(response, value) {
  response.writeHead(200, { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" });
  response.end(JSON.stringify(value));
}
const server = createServer((request, response) => {
  const url = new URL(request.url || "/", `http://${HOST}`);
  if (request.method !== "GET") { response.writeHead(405, { Allow: "GET" }); response.end("Method Not Allowed"); return; }
  if (url.pathname === "/") {
    try { response.writeHead(200, { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'" }); response.end(readFileSync(HTML_PATH)); }
    catch { response.writeHead(500); response.end("Viewer unavailable"); }
    return;
  }
  if (url.pathname === "/api/state") { json(response, snapshot()); return; }
  if (url.pathname === "/events") {
    response.writeHead(200, { "Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache", Connection: "keep-alive", "X-Content-Type-Options": "nosniff" });
    response.write(`data: ${JSON.stringify(snapshot())}\n\n`); clients.add(response); request.on("close", () => clients.delete(response)); return;
  }
  response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8", "X-Content-Type-Options": "nosniff" }); response.end("Not found");
});

if (!existsSync(TEAM_CHAT_DIR)) { try { mkdirSync(TEAM_CHAT_DIR, { recursive: true }); } catch {} }
try { watch(TEAM_CHAT_DIR, { persistent: false }, notify); } catch {}
setInterval(notify, 700).unref();
server.listen(PORT, HOST, () => console.log(`Workflow viewer listening at http://${HOST}:${PORT}`));
