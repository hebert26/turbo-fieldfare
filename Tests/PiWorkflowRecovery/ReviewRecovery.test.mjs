import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import teamChatExtension, { __test__ as extensionTest } from "/Users/dev-machine/.pi/agent/extensions/team-chat/index.ts";
import { deriveTeamState, isCurrentEscalationRequest } from "/Users/dev-machine/.pi/agent/extensions/team-chat/workflow.ts";

const TEAM_ID = "workflow-recovery-fixture";
const roster = [
  { name: "impl", duties: ["implementation"], work: "Implement the approved change" },
  { name: "terra-review", duties: ["plan_review", "initial_review"], work: "Review independently" },
  { name: "luna-review", duties: ["initial_review"], work: "Review independently" },
  { name: "verify", duties: ["verification"], work: "Verify the evidence" },
  { name: "terra-high", duties: ["escalation"], work: "Resolve genuine review disagreement" },
];

function event(type, sender, extra = {}) {
  return { type, team_id: TEAM_ID, sender, ts: 1, ...extra };
}

function indexed(events) {
  return events.map((entry, index) => ({ ...entry, index }));
}

function seed({ comparisons = [], requests = [], verdicts = [], extraRoster = roster } = {}) {
  const members = extraRoster.map((member) => ({ ...member, duties: [...member.duties] }));
  const events = [
    event("session_bound", "main", { session_id: "recovery-test" }),
    event("roster_set", "main", { members, goal: "Review recovery", mode: "implementation", revision: 1 }),
    ...members.filter((member) => !member.duties.includes("escalation")).map((member) => event("join", member.name)),
    event("join", "terra-high"),
  ];
  const splitIndex = events.push(event("split_proposed", "impl", { assignments: members.map(({ name, work }) => ({ name, work })) })) - 1;
  events.push(event("split_reviewed", "terra-review", { proposal_index: splitIndex, decision: "accept" }));
  const planIndex = events.push(event("plan_submitted", "impl", { plan: "Review provenance", evidence: "fixture", risk: "stale comparison", check: "focused test" })) - 1;
  events.push(event("plan_reviewed", "terra-review", { proposal_index: planIndex, decision: "approve" }));
  events.push(event("initial_review_submitted", "terra-review", { proposal_index: planIndex, findings: "terra", evidence: "fixture", risk: "none" }));
  events.push(event("initial_review_submitted", "luna-review", { proposal_index: planIndex, findings: "luna", evidence: "fixture", risk: "none" }));
  for (const comparison of comparisons) events.push(event("initial_review_compared", comparison.sender, comparison));
  for (const request of requests) events.push(event("escalation_requested", request.sender ?? "main", request));
  for (const verdict of verdicts) events.push(event("escalation_verdict", verdict.sender ?? "terra-high", verdict));
  return indexed(events);
}

function latest(events, type) {
  return [...events].reverse().find((entry) => entry.type === type);
}

function currentReviews(events) {
  const state = deriveTeamState(events);
  const plan = latest(state.workflow, "plan_submitted");
  return [...state.roster.values()]
    .filter((member) => member.duties.includes("initial_review"))
    .map((member) => latest(state.workflow.filter((entry) => entry.type === "initial_review_submitted" && entry.sender === member.name && entry.proposal_index === plan.index), "initial_review_submitted"))
    .map((entry) => entry.index)
    .sort((left, right) => left - right);
}

function currentReviewBySender(events, sender) {
  const state = deriveTeamState(events);
  const plan = latest(state.workflow, "plan_submitted");
  return latest(state.workflow.filter((entry) => entry.type === "initial_review_submitted" && entry.sender === sender && entry.proposal_index === plan.index), "initial_review_submitted").index;
}

const BASE = seed();
const [INITIAL_TERRA, INITIAL_LUNA] = currentReviews(BASE);
const INITIAL_PLAN = latest(BASE, "plan_submitted").index;
const OLD_PAIR = seed({
  comparisons: [
    { sender: "terra-review", proposal_index: INITIAL_LUNA, decision: "challenge", message: "old disagreement" },
    { sender: "luna-review", proposal_index: INITIAL_TERRA, decision: "challenge", message: "old disagreement" },
  ],
  requests: [{ initial_review_indexes: [INITIAL_TERRA, INITIAL_LUNA], reason: "old pair" }],
});
const OLD_REQUEST = latest(OLD_PAIR, "escalation_requested").index;

function append(events, type, sender, extra = {}) {
  const next = [...events, { ...event(type, sender, extra), index: events.length }];
  return next;
}

function compare(events, sender, peerIndex, decision = "accept", authorIndex) {
  return append(events, "initial_review_compared", sender, {
    proposal_index: peerIndex,
    decision,
    message: `${decision} fixture comparison`,
    ...(authorIndex === undefined ? {} : { author_review_index: authorIndex }),
  });
}

function replacement(events, sender) {
  return append(events, "initial_review_submitted", sender, {
    proposal_index: latest(events, "plan_submitted").index,
    findings: `${sender} replacement`,
    evidence: "replacement fixture",
    risk: "none",
  });
}

function completeTwoReviewerRound({ authorField = true, decision = "accept" } = {}) {
  let events = seed();
  const terra = currentReviews(events)[0];
  const luna = currentReviews(events)[1];
  events = compare(events, "terra-review", luna, decision, authorField ? terra : undefined);
  events = compare(events, "luna-review", terra, decision, authorField ? luna : undefined);
  return events;
}

// The exact mixed-renewal order from the incident: both old reviews disagreed, the old request was
// blocked, Luna renewed, and Terra accepted Luna's replacement. The old Luna challenge must not
// survive Luna's renewal merely because it still names Terra's unchanged review.
let p17 = seed({
  comparisons: [
    { sender: "terra-review", proposal_index: INITIAL_LUNA, decision: "challenge", message: "old disagreement" },
    { sender: "luna-review", proposal_index: INITIAL_TERRA, decision: "challenge", message: "old disagreement" },
  ],
  requests: [{ initial_review_indexes: [INITIAL_TERRA, INITIAL_LUNA], reason: "old pair" }],
  verdicts: [{ proposal_index: OLD_REQUEST, verdict: "block", evidence: "old pair blocked" }],
});
p17 = replacement(p17, "luna-review");
p17 = compare(p17, "terra-review", currentReviews(p17)[1], "accept");
const p17Current = currentReviews(p17);
const p17LunaReplacement = latest(p17, "initial_review_submitted").index;
assert.equal(p17.find((entry) => entry.type === "plan_submitted").index, INITIAL_PLAN, "fixture records the plan in journal order");
assert.deepEqual(p17Current, [INITIAL_TERRA, p17LunaReplacement], "the fixture exposes the replaced current-review pair");
assert.equal(deriveTeamState(p17).stage, "initial_review_peer_challenge", "renewal leaves an actionable comparison stage, not orphan escalation");
assert.equal(isCurrentEscalationRequest(p17.find((entry) => entry.type === "escalation_requested"), deriveTeamState(p17)), false, "the old request is historical after a renewal");
assert.deepEqual(
  extensionTest.deliveryEvidence([latest(p17, "initial_review_submitted")], deriveTeamState(p17)).map((entry) => entry.index),
  p17Current,
  "peer-stage delivery includes only the selected current review submissions",
);

// Renewing either reviewer first must invalidate the same old comparisons. Fresh directed accepts
// then advance normally, proving the repair is symmetric rather than tied to Luna's incident order.
for (const order of [
  ["terra-review", "luna-review"],
  ["luna-review", "terra-review"],
]) {
  let events = completeTwoReviewerRound({ authorField: false });
  for (const reviewer of order) events = replacement(events, reviewer);
  assert.equal(deriveTeamState(events).stage, "initial_review_peer_challenge", `${order.join(" then ")}: stale comparisons cannot approve renewed reviews`);
  const terra = currentReviewBySender(events, "terra-review");
  const luna = currentReviewBySender(events, "luna-review");
  events = compare(events, "terra-review", luna, "accept", terra);
  assert.equal(deriveTeamState(events).stage, "initial_review_peer_challenge", `${order.join(" then ")}: both directed comparisons remain required`);
  events = compare(events, "luna-review", terra, "accept", luna);
  assert.equal(deriveTeamState(events).stage, "implementation", `${order.join(" then ")}: fresh all-accept comparisons restore normal implementation`);
}

// A current disagreement requires an exact current-pair escalation request. An old request or old
// block cannot authorize it, and a current block leaves the team in corrective review until a new
// pair of comparisons is completed.
let currentDisagreement = seed();
currentDisagreement = compare(currentDisagreement, "terra-review", currentReviews(currentDisagreement)[1], "accept", currentReviews(currentDisagreement)[0]);
currentDisagreement = compare(currentDisagreement, "luna-review", currentReviews(currentDisagreement)[0], "challenge", currentReviews(currentDisagreement)[1]);
const disagreementPair = currentReviews(currentDisagreement);
currentDisagreement = append(currentDisagreement, "escalation_requested", "luna-review", { initial_review_indexes: disagreementPair, reason: "current disagreement" });
const currentRequest = latest(currentDisagreement, "escalation_requested");
assert.equal(deriveTeamState(currentDisagreement).stage, "escalation", "a genuine current disagreement enters escalation");
assert.equal(isCurrentEscalationRequest(currentRequest, deriveTeamState(currentDisagreement)), true, "the current request is actionable");
currentDisagreement = append(currentDisagreement, "escalation_verdict", "terra-high", { proposal_index: currentRequest.index, verdict: "block", evidence: "current pair blocked" });
assert.equal(deriveTeamState(currentDisagreement).stage, "initial_review_resolution", "a current block cannot become approval");
currentDisagreement = replacement(currentDisagreement, "luna-review");
assert.equal(deriveTeamState(currentDisagreement).stage, "initial_review_peer_challenge", "a corrective review alone does not approve the blocked work");

const [terraAfterBlock, lunaAfterBlock] = currentReviews(currentDisagreement);
currentDisagreement = compare(currentDisagreement, "terra-review", lunaAfterBlock, "accept", terraAfterBlock);
currentDisagreement = compare(currentDisagreement, "luna-review", terraAfterBlock, "accept", lunaAfterBlock);
assert.equal(deriveTeamState(currentDisagreement).stage, "implementation", "normal corrective reviews are required after a block");

// One reviewer completes without comparisons; two reviewers require both directed comparisons; three
// reviewers require all six directed comparisons, each tied to the author's current review.
const oneReviewer = seed({ extraRoster: [roster[0], roster[1], roster[3], roster[4]] });
assert.equal(deriveTeamState(oneReviewer).stage, "implementation", "one initial reviewer needs no peer comparison");
const twoReviewer = completeTwoReviewerRound();
assert.equal(deriveTeamState(twoReviewer).stage, "implementation", "two reviewers need both directed comparisons");
const threeRoster = [
  roster[0],
  roster[1],
  roster[2],
  { name: "third-review", duties: ["initial_review"], work: "Review independently" },
  roster[3],
  roster[4],
];
let threeReviewer = seed({ extraRoster: threeRoster });
threeReviewer = replacement(threeReviewer, "third-review");
const threeCurrent = currentReviews(threeReviewer);
const authorIndexes = new Map([
  ["terra-review", threeCurrent[0]],
  ["luna-review", threeCurrent[1]],
  ["third-review", threeCurrent[2]],
]);
for (const [author, authorIndex] of authorIndexes) {
  for (const peer of threeCurrent) {
    if (peer !== authorIndex) threeReviewer = compare(threeReviewer, author, peer, "accept", authorIndex);
  }
}
assert.equal(deriveTeamState(threeReviewer).stage, "implementation", "three reviewers require every directed current pair");

// Tool-level checks use only an isolated journal and the registered public team tools. They verify
// the new author provenance field and that stale/self/unauthorized proposals still fail closed.
const isolated = mkdtempSync(join(tmpdir(), "pi-workflow-recovery-"));
const oldTeamDir = process.env.PI_TEAM_CHAT_DIR;
const oldName = process.env.PI_SUBAGENT_NAME;
const oldAgent = process.env.PI_SUBAGENT_AGENT;
const tools = new Map();
const fakeEvents = { on() {} };
teamChatExtension({
  events: fakeEvents,
  on() {},
  registerTool(tool) { tools.set(tool.name, tool); },
  registerCommand() {},
  registerMessageRenderer() {},
  sendMessage() {},
  appendEntry() {},
});
const ctx = {
  cwd: process.cwd(),
  hasUI: false,
  model: { provider: "openai-codex", id: "gpt-5.6-terra" },
  thinkingLevel: "high",
  ui: { notify() {}, setStatus() {}, setWidget() {} },
  sessionManager: { getSessionId: () => "recovery-test", getSessionFile: () => join(isolated, "session.jsonl") },
};
const invoke = async (toolName, sender, params) => {
  process.env.PI_SUBAGENT_NAME = sender === "main" ? "" : sender;
  delete process.env.PI_SUBAGENT_AGENT;
  return tools.get(toolName).execute("recovery-test", params, undefined, undefined, ctx);
};
try {
  process.env.PI_TEAM_CHAT_DIR = isolated;
  const liveSeed = seed({
    comparisons: [
      { sender: "terra-review", proposal_index: INITIAL_LUNA, decision: "challenge", message: "old disagreement" },
      { sender: "luna-review", proposal_index: INITIAL_TERRA, decision: "challenge", message: "old disagreement" },
    ],
    requests: [{ initial_review_indexes: [INITIAL_TERRA, INITIAL_LUNA], reason: "old pair" }],
    verdicts: [{ proposal_index: OLD_REQUEST, verdict: "block", evidence: "old pair blocked" }],
  });
  // Use the actual journal order for tool proposal indexes.
  const journal = liveSeed.map((entry) => JSON.stringify(entry)).join("\n") + "\n";
  writeFileSync(join(isolated, `${TEAM_ID}.jsonl`), journal);
  const beforeBytes = readFileSync(join(isolated, `${TEAM_ID}.jsonl`));
  const liveState = deriveTeamState(liveSeed);
  assert.equal(liveState.stage, "initial_review_resolution");

  const replacementResult = await invoke("team_submit_initial_review", "luna-review", {
    team_id: TEAM_ID,
    proposal_index: INITIAL_PLAN,
    findings: "current Luna replacement",
    evidence: "isolated fixture",
    risk: "none",
  });
  const replacementEvent = replacementResult.details;
  assert.equal(replacementEvent.type, "initial_review_submitted");
  const afterReplacement = deriveTeamState(readJournal(isolated, TEAM_ID));
  const [terraLive, lunaLive] = currentReviews(readJournal(isolated, TEAM_ID));
  assert.equal(afterReplacement.stage, "initial_review_peer_challenge");
  await assert.rejects(
    invoke("team_compare_initial_review", "outsider", { team_id: TEAM_ID, proposal_index: terraLive, decision: "accept", message: "unauthorized" }),
    /outsider must be an active joined member of this team/,
    "an unassigned caller is rejected while peer comparison is actionable",
  );
  const terraCompare = await invoke("team_compare_initial_review", "terra-review", {
    team_id: TEAM_ID,
    proposal_index: lunaLive,
    decision: "accept",
    message: "current Luna review accepted",
  });
  assert.equal(terraCompare.details.comparison.author_review_index, terraLive, "the first renewed comparison records Terra's current review");
  assert.equal(deriveTeamState(readJournal(isolated, TEAM_ID)).stage, "initial_review_peer_challenge");
  await assert.rejects(
    invoke("team_compare_initial_review", "luna-review", { team_id: TEAM_ID, proposal_index: currentReviews(liveSeed)[1], decision: "accept", message: "stale or own" }),
    /Compare the current initial review of another assigned reviewer\. Your own review, an unknown index, and a stale round index are all rejected\./,
    "the stale old peer index is rejected after renewal",
  );
  const compareResult = await invoke("team_compare_initial_review", "luna-review", {
    team_id: TEAM_ID,
    proposal_index: terraLive,
    decision: "challenge",
    message: "current disagreement",
  });
  assert.equal(compareResult.details.comparison.author_review_index, lunaLive, "new comparisons record the author's current review index");
  assert.equal(compareResult.details.escalation.initial_review_indexes.join(","), [terraLive, lunaLive].sort((a, b) => a - b).join(","));
  const request = compareResult.details.escalation;
  assert.equal(deriveTeamState(readJournal(isolated, TEAM_ID)).stage, "escalation");

  await assert.rejects(
    invoke("team_submit_escalation_verdict", "terra-high", { team_id: TEAM_ID, proposal_index: request.index - 1, verdict: "proceed", evidence: "wrong request" }),
    /Review the current proposal from team_read_messages before deciding\. The proposal index is missing or stale\./,
    "a verdict cannot target a historical escalation request",
  );
  await invoke("team_submit_escalation_verdict", "terra-high", { team_id: TEAM_ID, proposal_index: request.index, verdict: "block", evidence: "current pair blocked" });
  assert.equal(deriveTeamState(readJournal(isolated, TEAM_ID)).stage, "initial_review_resolution");

  // A seeded legacy journal is read-only compatibility input. Derivation and guard failures never
  // rewrite its bytes or renumber its historical events.
  const afterBytes = readFileSync(join(isolated, `${TEAM_ID}.jsonl`));
  const beforeHash = createHash("sha256").update(beforeBytes).digest("hex");
  assert.equal(createHash("sha256").update(afterBytes.subarray(0, beforeBytes.length)).digest("hex"), beforeHash, "seeded legacy bytes remain unchanged as new events append");
} finally {
  if (oldTeamDir === undefined) delete process.env.PI_TEAM_CHAT_DIR;
  else process.env.PI_TEAM_CHAT_DIR = oldTeamDir;
  if (oldName === undefined) delete process.env.PI_SUBAGENT_NAME;
  else process.env.PI_SUBAGENT_NAME = oldName;
  if (oldAgent === undefined) delete process.env.PI_SUBAGENT_AGENT;
  else process.env.PI_SUBAGENT_AGENT = oldAgent;
  rmSync(isolated, { recursive: true, force: true });
}

function readJournal(dir, teamId) {
  return readFileSync(join(dir, `${teamId}.jsonl`), "utf8").trim().split("\n").map(JSON.parse);
}

// Legacy chronology remains valid before an author's renewal, then expires after it without rewriting history.
const legacyBeforeRenewal = completeTwoReviewerRound({ authorField: false });
assert.equal(deriveTeamState(legacyBeforeRenewal).stage, "implementation");
const legacyAfterRenewal = replacement(legacyBeforeRenewal, "luna-review");
assert.equal(deriveTeamState(legacyAfterRenewal).stage, "initial_review_peer_challenge");

console.log("workflow recovery regression cases passed");
