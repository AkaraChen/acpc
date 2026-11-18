//! Reduces one ACP prompt turn into a Codex `Turn` and its live notifications.
//!
//! The same reducer serves live turns and `session/load` replay; replay
//! simply discards the notifications and keeps the items.

use std::collections::HashMap;
use std::path::PathBuf;

use acp_client::schema as acp;
use codex_app_server_protocol::AgentMessageDeltaNotification;
use codex_app_server_protocol::CommandExecutionOutputDeltaNotification;
use codex_app_server_protocol::DynamicToolCallOutputContentItem;
use codex_app_server_protocol::FileChangePatchUpdatedNotification;
use codex_app_server_protocol::ItemCompletedNotification;
use codex_app_server_protocol::ItemStartedNotification;
use codex_app_server_protocol::ReasoningSummaryTextDeltaNotification;
use codex_app_server_protocol::ServerNotification;
use codex_app_server_protocol::ThreadItem;
use codex_app_server_protocol::ThreadItemPayload;
use codex_app_server_protocol::Turn;
use codex_app_server_protocol::TurnError;
use codex_app_server_protocol::TurnPlanStep;
use codex_app_server_protocol::TurnPlanStepStatus;
use codex_app_server_protocol::TurnPlanUpdatedNotification;
use codex_app_server_protocol::TurnStatus;
use codex_app_server_protocol::UserInput;

use crate::content::display_text;
use crate::records;
use crate::tool_calls;
use crate::tool_calls::ItemKind;
use crate::tool_calls::Outcome;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextKind {
    Message,
    Reasoning,
}

struct OpenText {
    kind: TextKind,
    index: usize,
    message_id: Option<String>,
    text: String,
}

struct ToolEntry {
    index: usize,
    kind: ItemKind,
    call: acp::ToolCall,
    started_at_ms: i64,
    emitted_output: String,
    /// How the call ended, once it has.
    outcome: Option<Outcome>,
    /// `item/started` went out. A file change waits for its diff: the TUI
    /// draws the change (and its `+N -M`) once, from `item/started`.
    announced: bool,
    finished: bool,
}

/// A tool call a subagent made. claude-agent-acp sends a subagent's calls
/// and text in the parent session, tagged with the id of the Agent/Task call
/// that started it (`_meta.claudeCode.parentToolUseId`). They are folded into
/// that call's item as a progress line instead of becoming items of their own.
struct ChildCall {
    parent: String,
    call: acp::ToolCall,
}

pub(crate) struct TurnBuilder {
    pub(crate) id: String,
    thread_id: String,
    cwd: PathBuf,
    pub(crate) started_at_ms: i64,
    items: Vec<ThreadItem>,
    open_text: Option<OpenText>,
    tools: HashMap<String, ToolEntry>,
    /// Subagent tool calls by call id, in the order they started.
    children: Vec<(String, ChildCall)>,
    next_item: u64,
}

/// The Agent/Task call a subagent's update belongs to, from claude-agent-acp's
/// `_meta.claudeCode.parentToolUseId`.
fn parent_tool_use_id(meta: Option<&acp::Meta>) -> Option<String> {
    meta?
        .get("claudeCode")?
        .get("parentToolUseId")?
        .as_str()
        .filter(|id| !id.is_empty())
        .map(str::to_string)
}

pub(crate) type Emit = Vec<ServerNotification>;

impl TurnBuilder {
    pub(crate) fn new(id: String, thread_id: String, cwd: PathBuf, started_at_ms: i64) -> Self {
        Self {
            id,
            thread_id,
            cwd,
            started_at_ms,
            items: Vec::new(),
            open_text: None,
            tools: HashMap::new(),
            children: Vec::new(),
            next_item: 1,
        }
    }

    fn next_item_id(&mut self) -> String {
        let id = format!("{}-{}", self.id, self.next_item);
        self.next_item += 1;
        id
    }

    pub(crate) fn has_items(&self) -> bool {
        !self.items.is_empty()
    }

    fn started(&self, index: usize, emit: &mut Emit) {
        emit.push(ServerNotification::ItemStarted(ItemStartedNotification {
            item: self.items[index].clone(),
            thread_id: self.thread_id.clone(),
            turn_id: self.id.clone(),
            started_at_ms: self.items[index]
                .started_at_ms
                .unwrap_or(self.started_at_ms),
        }));
    }

    fn completed(&mut self, index: usize, emit: &mut Emit) {
        let now = records::now_ms();
        self.items[index].completed_at_ms = Some(now);
        emit.push(ServerNotification::ItemCompleted(
            ItemCompletedNotification {
                item: self.items[index].clone(),
                thread_id: self.thread_id.clone(),
                turn_id: self.id.clone(),
                content_token_count: None,
                completed_at_ms: now,
            },
        ));
    }

    /// A user prompt replayed by `session/load`.
    pub(crate) fn user_chunk(&mut self, chunk: &acp::ContentChunk, emit: &mut Emit) {
        self.close_text(emit);
        let Some(input) = crate::content::user_input(&chunk.content) else {
            return;
        };
        // Consecutive chunks of one user message extend the same item.
        if let Some(ThreadItem {
            payload: ThreadItemPayload::UserMessage { content, .. },
            ..
        }) = self.items.last_mut()
        {
            match (content.last_mut(), &input) {
                (Some(UserInput::Text { text, .. }), UserInput::Text { text: more, .. }) => {
                    text.push_str(more)
                }
                _ => content.push(input),
            }
            return;
        }
        let id = self.next_item_id();
        let now = records::now_ms();
        let mut item = records::item(
            ThreadItemPayload::UserMessage {
                id,
                client_id: None,
                content: vec![input],
            },
            now,
        );
        item.completed_at_ms = Some(now);
        self.items.push(item);
    }

    pub(crate) fn has_agent_output(&self) -> bool {
        self.items
            .iter()
            .any(|item| !matches!(item.payload, ThreadItemPayload::UserMessage { .. }))
    }

    pub(crate) fn agent_message(&mut self, chunk: &acp::ContentChunk, emit: &mut Emit) {
        // A subagent's text is not the answer; its report is the call's output.
        if self.is_subagent_update(chunk.meta.as_ref()) {
            return;
        }
        self.text(TextKind::Message, chunk, emit);
    }

    pub(crate) fn agent_thought(&mut self, chunk: &acp::ContentChunk, emit: &mut Emit) {
        if self.is_subagent_update(chunk.meta.as_ref()) {
            return;
        }
        self.text(TextKind::Reasoning, chunk, emit);
    }

    /// Whether an update comes from a subagent whose Agent call is running.
    fn is_subagent_update(&self, meta: Option<&acp::Meta>) -> bool {
        self.subagent_parent(meta).is_some()
    }

    /// The running Agent call that a subagent's update belongs to.
    fn subagent_parent(&self, meta: Option<&acp::Meta>) -> Option<String> {
        parent_tool_use_id(meta).filter(|parent| self.running_parent(parent))
    }

    /// Whether `key` is a running tool item that can show subagent progress.
    fn running_parent(&self, key: &str) -> bool {
        self.tools
            .get(key)
            .is_some_and(|entry| entry.kind == ItemKind::Dynamic && !entry.finished)
    }

    fn child_index(&self, key: &str) -> Option<usize> {
        self.children.iter().position(|(child, _)| child == key)
    }

    /// Record a subagent's tool call against its running Agent call `parent`.
    fn fold_child(&mut self, parent: String, call: acp::ToolCall, emit: &mut Emit) {
        let key = call.tool_call_id.to_string();
        self.children.push((
            key,
            ChildCall {
                parent: parent.clone(),
                call,
            },
        ));
        self.show_progress(&parent, emit);
    }

    /// What the subagents of call `parent` have done, for its item.
    fn progress(&self, parent: &str) -> Option<String> {
        let calls = self
            .children
            .iter()
            .filter(|(_, child)| child.parent == parent)
            .map(|(_, child)| &child.call)
            .collect::<Vec<_>>();
        let latest = calls.last()?;
        let running = calls
            .iter()
            .rev()
            .find(|call| tool_calls::outcome(call) == Outcome::Running);
        let count = match calls.len() {
            1 => "1 tool call".to_string(),
            n => format!("{n} tool calls"),
        };
        Some(match running {
            Some(call) => format!("{count} · now: {}", call.title),
            None => format!("{count} · last: {}", latest.title),
        })
    }

    /// The payload of a running tool item. A dynamic item carries its
    /// subagents' progress as output, so the TUI shows it under the call.
    fn running_payload(&self, key: &str) -> Option<ThreadItemPayload> {
        let entry = self.tools.get(key)?;
        let item_id = item_id(&self.items[entry.index].payload).to_string();
        let mut payload = tool_calls::payload(
            &item_id,
            entry.kind,
            &entry.call,
            Outcome::Running,
            &self.cwd,
            None,
        );
        if let (Some(progress), ThreadItemPayload::DynamicToolCall { content_items, .. }) =
            (self.progress(key), &mut payload)
        {
            *content_items = Some(vec![DynamicToolCallOutputContentItem::InputText {
                text: progress,
            }]);
        }
        Some(payload)
    }

    /// Re-announce a running Agent call with its subagents' latest progress;
    /// the TUI updates the call's cell in place.
    fn show_progress(&mut self, parent: &str, emit: &mut Emit) {
        let Some(payload) = self.running_payload(parent) else {
            return;
        };
        let index = self.tools[parent].index;
        self.items[index].payload = payload;
        if self.tools[parent].announced {
            self.started(index, emit);
        }
    }

    fn text(&mut self, kind: TextKind, chunk: &acp::ContentChunk, emit: &mut Emit) {
        let delta = display_text(&chunk.content);
        let message_id = chunk.message_id.as_ref().map(ToString::to_string);
        let continues = self.open_text.as_ref().is_some_and(|open| {
            open.kind == kind
                && (message_id.is_none()
                    || open.message_id.is_none()
                    || open.message_id == message_id)
        });
        if !continues {
            self.close_text(emit);
            let id = self.next_item_id();
            let payload = match kind {
                TextKind::Message => ThreadItemPayload::AgentMessage {
                    moderation: None,
                    id,
                    text: String::new(),
                    phase: None,
                    memory_citation: None,
                    delivery: None,
                    questions: None,
                },
                TextKind::Reasoning => ThreadItemPayload::Reasoning {
                    id,
                    summary: Vec::new(),
                    content: Vec::new(),
                },
            };
            self.items.push(records::item(payload, records::now_ms()));
            let index = self.items.len() - 1;
            self.started(index, emit);
            self.open_text = Some(OpenText {
                kind,
                index,
                message_id: message_id.clone(),
                text: String::new(),
            });
        }
        let Some(open) = self.open_text.as_mut() else {
            return;
        };
        if open.message_id.is_none() {
            open.message_id = message_id;
        }
        if delta.is_empty() {
            return;
        }
        open.text.push_str(&delta);
        let item_id = item_id(&self.items[open.index].payload).to_string();
        match open.kind {
            TextKind::Message => emit.push(ServerNotification::AgentMessageDelta(
                AgentMessageDeltaNotification {
                    thread_id: self.thread_id.clone(),
                    turn_id: self.id.clone(),
                    item_id,
                    delta,
                },
            )),
            TextKind::Reasoning => emit.push(ServerNotification::ReasoningSummaryTextDelta(
                ReasoningSummaryTextDeltaNotification {
                    thread_id: self.thread_id.clone(),
                    turn_id: self.id.clone(),
                    item_id,
                    delta,
                    summary_index: 0,
                },
            )),
        }
    }

    fn close_text(&mut self, emit: &mut Emit) {
        let Some(open) = self.open_text.take() else {
            return;
        };
        match &mut self.items[open.index].payload {
            ThreadItemPayload::AgentMessage { text, .. } => *text = open.text,
            ThreadItemPayload::Reasoning { summary, .. } if !open.text.is_empty() => {
                *summary = vec![open.text];
            }
            _ => {}
        }
        self.completed(open.index, emit);
    }

    /// The item id assigned to an ACP tool call, creating the item if this is
    /// the first time the call is seen.
    pub(crate) fn tool_item(
        &mut self,
        update: &acp::ToolCallUpdate,
        emit: &mut Emit,
    ) -> (String, ItemKind) {
        let key = update.tool_call_id.to_string();
        if let Some(index) = self.child_index(&key) {
            // A subagent's call that needs the user (a permission request)
            // becomes an item of its own, so the prompt has a call to show.
            let (_, mut child) = self.children.remove(index);
            tool_calls::apply_update(&mut child.call, update.fields.clone());
            if self.running_parent(&child.parent) {
                self.show_progress(&child.parent, emit);
            }
            self.add_tool_call(child.call, emit);
        } else if !self.tools.contains_key(&key) {
            self.add_tool_call(tool_calls::from_update(update), emit);
        } else {
            self.tool_call_update(update.clone(), emit);
        }
        // The approval shows the change, so announce it now, with diffs read
        // against the files the tool has not touched yet.
        if let Some(entry) = self.tools.get_mut(&key)
            && entry.kind == ItemKind::FileChange
            && !entry.finished
        {
            tool_calls::resolve_against_files(&mut entry.call, &self.cwd);
            self.refresh_file_change(&key, None, emit);
            self.announce(&key, emit);
        }
        let entry = &self.tools[&key];
        (
            item_id(&self.items[entry.index].payload).to_string(),
            entry.kind,
        )
    }

    /// Emit `item/started` for a tool item that has not had it yet.
    fn announce(&mut self, key: &str, emit: &mut Emit) {
        let Some(entry) = self.tools.get_mut(key) else {
            return;
        };
        if entry.announced {
            return;
        }
        entry.announced = true;
        let index = entry.index;
        self.started(index, emit);
    }

    /// Rebuild a file change item from its call, telling the client when an
    /// announced item's changes moved.
    fn refresh_file_change(&mut self, key: &str, outcome: Option<Outcome>, emit: &mut Emit) {
        let Some(entry) = self.tools.get(key) else {
            return;
        };
        let index = entry.index;
        let item_id = item_id(&self.items[index].payload).to_string();
        let payload = tool_calls::payload(
            &item_id,
            ItemKind::FileChange,
            &entry.call,
            outcome.unwrap_or(Outcome::Running),
            &self.cwd,
            None,
        );
        let changed = match (&self.items[index].payload, &payload) {
            (
                ThreadItemPayload::FileChange {
                    changes: before, ..
                },
                ThreadItemPayload::FileChange { changes: after, .. },
            ) => before != after,
            _ => true,
        };
        if changed
            && entry.announced
            && let ThreadItemPayload::FileChange { changes, .. } = &payload
        {
            emit.push(ServerNotification::FileChangePatchUpdated(
                FileChangePatchUpdatedNotification {
                    thread_id: self.thread_id.clone(),
                    turn_id: self.id.clone(),
                    item_id,
                    changes: changes.clone(),
                },
            ));
        }
        match (&mut self.items[index].payload, payload) {
            // A finished item keeps its outcome and duration.
            (
                ThreadItemPayload::FileChange { changes, .. },
                ThreadItemPayload::FileChange { changes: fresh, .. },
            ) if outcome.is_some() => *changes = fresh,
            (slot, payload) => *slot = payload,
        }
    }

    pub(crate) fn tool_call(&mut self, call: acp::ToolCall, emit: &mut Emit) {
        let key = call.tool_call_id.to_string();
        if self.child_index(&key).is_none()
            && !self.tools.contains_key(&key)
            && let Some(parent) = self.subagent_parent(call.meta.as_ref())
        {
            self.fold_child(parent, call, emit);
            return;
        }
        self.add_tool_call(call, emit);
    }

    fn add_tool_call(&mut self, call: acp::ToolCall, emit: &mut Emit) {
        let key = call.tool_call_id.to_string();
        if self.tools.contains_key(&key) || self.child_index(&key).is_some() {
            // Agents may re-announce a call; treat it as a full update.
            let fields = acp::ToolCallUpdateFields::new()
                .kind(call.kind)
                .status(call.status)
                .title(call.title.clone())
                .content(call.content.clone())
                .locations(call.locations.clone())
                .raw_input(call.raw_input.clone())
                .raw_output(call.raw_output.clone());
            self.tool_call_update(acp::ToolCallUpdate::new(call.tool_call_id, fields), emit);
            return;
        }
        self.close_text(emit);
        let kind = tool_calls::classify(&call);
        let id = self.next_item_id();
        let now = records::now_ms();
        let payload = tool_calls::payload(&id, kind, &call, Outcome::Running, &self.cwd, None);
        self.items.push(records::item(payload, now));
        let index = self.items.len() - 1;
        // Agents announce an edit before its diff is known (claude-agent-acp
        // sends "Edit" with no content while the input streams), so a file
        // change starts at its permission request or when it finishes.
        let announced = kind != ItemKind::FileChange;
        if announced {
            self.started(index, emit);
        }
        let output = tool_calls::output_text(&call);
        let finished = tool_calls::outcome(&call) != Outcome::Running;
        self.tools.insert(
            key.clone(),
            ToolEntry {
                index,
                kind,
                call,
                started_at_ms: now,
                emitted_output: String::new(),
                outcome: None,
                announced,
                finished: false,
            },
        );
        if kind == ItemKind::Command && !output.is_empty() {
            self.emit_output(&key, emit);
        }
        if finished {
            self.finish_tool(&key, None, emit);
        }
    }

    pub(crate) fn tool_call_update(&mut self, update: acp::ToolCallUpdate, emit: &mut Emit) {
        let key = update.tool_call_id.to_string();
        if let Some(index) = self.child_index(&key) {
            // Later updates of a subagent's call can lose the parent tag.
            let (_, child) = &mut self.children[index];
            tool_calls::apply_update(&mut child.call, update.fields);
            let parent = child.parent.clone();
            if self.running_parent(&parent) {
                self.show_progress(&parent, emit);
            }
            return;
        }
        let Some(entry) = self.tools.get_mut(&key) else {
            // An update for an unannounced call: start it from what we know.
            let call = tool_calls::from_update(&update);
            self.tool_call(call, emit);
            return;
        };
        if entry.finished {
            // claude-agent-acp sends the final diff of an edit from a hook
            // that can run after the call completed; keep the item current.
            if entry.kind == ItemKind::FileChange && update.fields.content.is_some() {
                let outcome = entry.outcome;
                let mut fields = update.fields;
                fields.status = None;
                tool_calls::apply_update(&mut entry.call, fields);
                self.refresh_file_change(&key, outcome, emit);
            }
            return;
        }
        tool_calls::apply_update(&mut entry.call, update.fields);
        let kind = entry.kind;
        let finished = tool_calls::outcome(&entry.call) != Outcome::Running;
        match kind {
            ItemKind::Command | ItemKind::Read | ItemKind::Dynamic => {
                // Keep the stored item current (an approval prompt shows it).
                let index = self.tools[&key].index;
                if let Some(payload) = self.running_payload(&key) {
                    self.items[index].payload = payload;
                }
                if kind == ItemKind::Command {
                    self.emit_output(&key, emit);
                }
            }
            ItemKind::FileChange => self.refresh_file_change(&key, None, emit),
        }
        if finished {
            self.finish_tool(&key, None, emit);
        }
    }

    fn emit_output(&mut self, key: &str, emit: &mut Emit) {
        let Some(entry) = self.tools.get_mut(key) else {
            return;
        };
        let output = tool_calls::output_text(&entry.call);
        let delta = match output.strip_prefix(entry.emitted_output.as_str()) {
            Some(delta) => delta.to_string(),
            // Output was rewritten; the completed item carries the full text.
            None => String::new(),
        };
        entry.emitted_output = output;
        if delta.is_empty() {
            return;
        }
        let item_id = item_id(&self.items[entry.index].payload).to_string();
        emit.push(ServerNotification::CommandExecutionOutputDelta(
            CommandExecutionOutputDeltaNotification {
                thread_id: self.thread_id.clone(),
                turn_id: self.id.clone(),
                item_id,
                delta,
            },
        ));
    }

    /// Complete a tool item. `forced` overrides the agent-reported outcome
    /// (used for declined permissions and turn cancellation).
    pub(crate) fn finish_tool(&mut self, key: &str, forced: Option<Outcome>, emit: &mut Emit) {
        let Some(entry) = self.tools.get_mut(key) else {
            return;
        };
        if entry.finished {
            return;
        }
        entry.finished = true;
        let outcome = forced.unwrap_or_else(|| match tool_calls::outcome(&entry.call) {
            Outcome::Running => Outcome::Failed,
            outcome => outcome,
        });
        entry.outcome = Some(outcome);
        if !entry.announced {
            // A file change starts here with its final changes.
            if entry.kind == ItemKind::FileChange {
                self.refresh_file_change(key, None, emit);
            }
            self.announce(key, emit);
        }
        let Some(entry) = self.tools.get_mut(key) else {
            return;
        };
        let duration = records::now_ms().saturating_sub(entry.started_at_ms);
        let index = entry.index;
        let item_id = item_id(&self.items[index].payload).to_string();
        self.items[index].payload = tool_calls::payload(
            &item_id,
            entry.kind,
            &entry.call,
            outcome,
            &self.cwd,
            Some(duration),
        );
        self.completed(index, emit);
    }

    /// The agent's current title for a tool call.
    pub(crate) fn tool_title(&self, tool_call_id: &acp::ToolCallId) -> Option<String> {
        self.tools
            .get(&tool_call_id.to_string())
            .map(|entry| entry.call.title.clone())
            .filter(|title| !title.is_empty())
    }

    pub(crate) fn item_payload(&self, wanted: &str) -> Option<ThreadItemPayload> {
        self.items
            .iter()
            .find(|item| item_id(&item.payload) == wanted)
            .map(|item| item.payload.clone())
    }

    pub(crate) fn plan(&self, plan: &acp::Plan, emit: &mut Emit) {
        emit.push(ServerNotification::TurnPlanUpdated(
            TurnPlanUpdatedNotification {
                thread_id: self.thread_id.clone(),
                turn_id: self.id.clone(),
                explanation: None,
                plan: plan
                    .entries
                    .iter()
                    .map(|entry| TurnPlanStep {
                        step: entry.content.clone(),
                        status: match entry.status {
                            acp::PlanEntryStatus::Completed => TurnPlanStepStatus::Completed,
                            acp::PlanEntryStatus::InProgress => TurnPlanStepStatus::InProgress,
                            _ => TurnPlanStepStatus::Pending,
                        },
                    })
                    .collect(),
            },
        ));
    }

    /// A context compaction reported by the agent (`compaction_update`).
    pub(crate) fn compaction(&mut self, update: &acp::CompactionUpdate, emit: &mut Emit) {
        let key = format!("compaction:{}", update.compaction_id);
        if let Some(entry) = self.tools.get(&key) {
            if !matches!(update.status, acp::CompactionStatus::InProgress) && !entry.finished {
                let index = entry.index;
                if let Some(entry) = self.tools.get_mut(&key) {
                    entry.finished = true;
                }
                self.completed(index, emit);
            }
            return;
        }
        self.close_text(emit);
        let id = self.next_item_id();
        self.items.push(records::item(
            ThreadItemPayload::ContextCompaction { id },
            records::now_ms(),
        ));
        let index = self.items.len() - 1;
        self.started(index, emit);
        let finished = !matches!(update.status, acp::CompactionStatus::InProgress);
        self.tools.insert(
            key,
            ToolEntry {
                index,
                kind: ItemKind::Dynamic,
                call: acp::ToolCall::new("compaction", "Compaction"),
                started_at_ms: records::now_ms(),
                emitted_output: String::new(),
                outcome: None,
                announced: true,
                finished,
            },
        );
        if finished {
            self.completed(index, emit);
        }
    }

    /// The turn so far, for a client attaching while it runs.
    pub(crate) fn snapshot(&self) -> Turn {
        let mut items = self.items.clone();
        if let Some(open) = &self.open_text {
            match &mut items[open.index].payload {
                ThreadItemPayload::AgentMessage { text, .. } => text.clone_from(&open.text),
                ThreadItemPayload::Reasoning { summary, .. } => {
                    *summary = vec![open.text.clone()];
                }
                _ => {}
            }
        }
        records::turn(
            &self.id,
            items,
            TurnStatus::InProgress,
            None,
            Some(self.started_at_ms),
            None,
        )
    }

    /// Close every open item and produce the finished turn.
    pub(crate) fn finish(
        mut self,
        status: TurnStatus,
        error: Option<TurnError>,
        emit: &mut Emit,
    ) -> Turn {
        self.close_text(emit);
        let cancelled = status == TurnStatus::Interrupted;
        let open: Vec<String> = self
            .tools
            .iter()
            .filter(|(key, entry)| !entry.finished && !key.starts_with("compaction:"))
            .map(|(key, _)| key.clone())
            .collect();
        for key in open {
            let forced = if cancelled {
                Outcome::Declined
            } else {
                Outcome::Failed
            };
            self.finish_tool(&key, Some(forced), emit);
        }
        let compactions: Vec<usize> = self
            .tools
            .iter()
            .filter(|(key, entry)| !entry.finished && key.starts_with("compaction:"))
            .map(|(_, entry)| entry.index)
            .collect();
        for index in compactions {
            self.completed(index, emit);
        }
        records::turn(
            &self.id,
            self.items,
            status,
            error,
            Some(self.started_at_ms),
            Some(records::now_ms()),
        )
    }
}

pub(crate) fn item_id(payload: &ThreadItemPayload) -> &str {
    match payload {
        ThreadItemPayload::UserMessage { id, .. }
        | ThreadItemPayload::AgentMessage { id, .. }
        | ThreadItemPayload::Reasoning { id, .. }
        | ThreadItemPayload::CommandExecution { id, .. }
        | ThreadItemPayload::FileChange { id, .. }
        | ThreadItemPayload::DynamicToolCall { id, .. }
        | ThreadItemPayload::ContextCompaction { id, .. } => id,
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn chunk(text: &str, message_id: Option<&str>) -> acp::ContentChunk {
        let chunk = acp::ContentChunk::new(acp::ContentBlock::Text(acp::TextContent::new(text)));
        match message_id {
            Some(id) => chunk.message_id(acp::MessageId::new(id)),
            None => chunk,
        }
    }

    fn methods(emit: &Emit) -> Vec<String> {
        emit.iter()
            .map(|n| {
                serde_json::to_value(n)
                    .ok()
                    .and_then(|v| v.get("method").and_then(|m| m.as_str()).map(str::to_string))
                    .unwrap_or_default()
            })
            .collect()
    }

    fn builder() -> TurnBuilder {
        TurnBuilder::new("t1".into(), "th".into(), "/w".into(), 0)
    }

    #[test]
    fn message_chunks_stream_into_one_item_until_interrupted_by_a_tool() {
        let mut turn = builder();
        let mut emit = Vec::new();
        turn.agent_thought(&chunk("hmm", None), &mut emit);
        turn.agent_message(&chunk("Hel", None), &mut emit);
        turn.agent_message(&chunk("lo", None), &mut emit);
        turn.tool_call(
            acp::ToolCall::new("c1", "ls").kind(acp::ToolKind::Execute),
            &mut emit,
        );
        turn.agent_message(&chunk("done", None), &mut emit);
        let finished = turn.finish(TurnStatus::Completed, None, &mut emit);
        assert_eq!(
            methods(&emit),
            vec![
                "item/started",
                "item/reasoning/summaryTextDelta",
                "item/completed",
                "item/started",
                "item/agentMessage/delta",
                "item/agentMessage/delta",
                "item/completed",
                "item/started",
                "item/started",
                "item/agentMessage/delta",
                "item/completed",
                "item/completed",
            ]
        );
        let texts: Vec<String> = finished
            .items
            .iter()
            .filter_map(|item| match &item.payload {
                ThreadItemPayload::AgentMessage { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["Hello".to_string(), "done".to_string()]);
    }

    #[test]
    fn changed_message_id_starts_a_new_message() {
        let mut turn = builder();
        let mut emit = Vec::new();
        turn.agent_message(&chunk("a", Some("m1")), &mut emit);
        turn.agent_message(&chunk("b", Some("m2")), &mut emit);
        let finished = turn.finish(TurnStatus::Completed, None, &mut emit);
        assert_eq!(finished.items.len(), 2);
    }

    #[test]
    fn cancelled_turn_declines_running_tools() {
        let mut turn = builder();
        let mut emit = Vec::new();
        turn.tool_call(
            acp::ToolCall::new("c1", "rm").kind(acp::ToolKind::Execute),
            &mut emit,
        );
        let finished = turn.finish(TurnStatus::Interrupted, None, &mut emit);
        assert!(matches!(
            finished.items[0].payload,
            ThreadItemPayload::CommandExecution {
                status: codex_app_server_protocol::CommandExecutionStatus::Declined,
                ..
            }
        ));
    }

    #[test]
    fn command_output_streams_as_deltas() {
        let mut turn = builder();
        let mut emit = Vec::new();
        turn.tool_call(
            acp::ToolCall::new("c1", "ls").kind(acp::ToolKind::Execute),
            &mut emit,
        );
        let text = |t: &str| {
            vec![acp::ToolCallContent::Content(acp::Content::new(
                acp::ContentBlock::Text(acp::TextContent::new(t)),
            ))]
        };
        turn.tool_call_update(
            acp::ToolCallUpdate::new("c1", acp::ToolCallUpdateFields::new().content(text("a\n"))),
            &mut emit,
        );
        turn.tool_call_update(
            acp::ToolCallUpdate::new(
                "c1",
                acp::ToolCallUpdateFields::new()
                    .content(text("a\nb\n"))
                    .status(acp::ToolCallStatus::Completed),
            ),
            &mut emit,
        );
        assert_eq!(
            methods(&emit),
            vec![
                "item/started",
                "item/commandExecution/outputDelta",
                "item/commandExecution/outputDelta",
                "item/completed",
            ]
        );
    }

    fn subagent_meta(parent: &str) -> acp::Meta {
        serde_json::from_value(serde_json::json!({"claudeCode": {"parentToolUseId": parent}}))
            .expect("meta")
    }

    fn agent_call() -> acp::ToolCall {
        acp::ToolCall::new("agent-1", "Explore auth flow").kind(acp::ToolKind::Think)
    }

    fn child_call(id: &'static str, title: &str) -> acp::ToolCall {
        let mut call = acp::ToolCall::new(id, title)
            .kind(acp::ToolKind::Read)
            .status(acp::ToolCallStatus::InProgress);
        call.meta = Some(subagent_meta("agent-1"));
        call
    }

    fn progress_of(turn: &TurnBuilder) -> Option<String> {
        match turn.item_payload("t1-1")? {
            ThreadItemPayload::DynamicToolCall {
                content_items: Some(items),
                ..
            } => items.into_iter().find_map(|item| match item {
                DynamicToolCallOutputContentItem::InputText { text } => Some(text),
                _ => None,
            }),
            _ => None,
        }
    }

    fn finish_update(id: &'static str) -> acp::ToolCallUpdate {
        acp::ToolCallUpdate::new(
            id,
            acp::ToolCallUpdateFields::new().status(acp::ToolCallStatus::Completed),
        )
    }

    #[test]
    fn subagent_calls_fold_into_the_agent_call() {
        let mut turn = builder();
        let mut emit = Emit::new();
        turn.tool_call(agent_call(), &mut emit);
        turn.tool_call(child_call("c1", "Read src/auth.rs"), &mut emit);
        turn.tool_call(child_call("c2", "Grep \"login\""), &mut emit);

        assert_eq!(turn.items.len(), 1, "children must not become items");
        assert_eq!(
            progress_of(&turn).as_deref(),
            Some("2 tool calls · now: Grep \"login\"")
        );
        // Each child re-announces the Agent call so the TUI cell updates.
        assert_eq!(methods(&emit), vec!["item/started"; 3]);

        let mut emit = Emit::new();
        turn.tool_call_update(finish_update("c2"), &mut emit);
        turn.tool_call_update(finish_update("c1"), &mut emit);
        assert_eq!(
            progress_of(&turn).as_deref(),
            Some("2 tool calls · last: Grep \"login\"")
        );
    }

    #[test]
    fn subagent_text_is_not_the_answer() {
        let mut turn = builder();
        let mut emit = Emit::new();
        turn.tool_call(agent_call(), &mut emit);
        let mut chunk = chunk("child says hi", None);
        chunk.meta = Some(subagent_meta("agent-1"));
        turn.agent_message(&chunk, &mut emit);
        turn.agent_thought(&chunk, &mut emit);
        assert_eq!(turn.items.len(), 1);
    }

    #[test]
    fn finished_agent_call_shows_its_report() {
        let mut turn = builder();
        let mut emit = Emit::new();
        turn.tool_call(agent_call(), &mut emit);
        turn.tool_call(child_call("c1", "Read src/auth.rs"), &mut emit);
        let report = acp::ToolCallUpdateFields::new()
            .status(acp::ToolCallStatus::Completed)
            .content(vec!["Auth lives in src/auth.rs".to_string().into()]);
        turn.tool_call_update(acp::ToolCallUpdate::new("agent-1", report), &mut emit);
        assert_eq!(
            progress_of(&turn).as_deref(),
            Some("Auth lives in src/auth.rs")
        );

        // A late child update after the Agent call ended changes nothing.
        let mut emit = Emit::new();
        turn.tool_call_update(finish_update("c1"), &mut emit);
        assert!(emit.is_empty());
        assert_eq!(turn.items.len(), 1);
    }

    #[test]
    fn calls_tagged_with_an_unknown_parent_stay_items() {
        let mut turn = builder();
        let mut emit = Emit::new();
        turn.tool_call(child_call("c1", "Read src/auth.rs"), &mut emit);
        assert_eq!(turn.items.len(), 1);
        assert!(matches!(
            turn.item_payload("t1-1"),
            Some(ThreadItemPayload::DynamicToolCall { tool, .. }) if tool == "Read src/auth.rs"
        ));
    }

    #[test]
    fn subagent_call_first_seen_as_an_update_folds_too() {
        let mut turn = builder();
        let mut emit = Emit::new();
        turn.tool_call(agent_call(), &mut emit);
        let mut update = acp::ToolCallUpdate::new(
            "c1",
            acp::ToolCallUpdateFields::new()
                .title("Read src/auth.rs".to_string())
                .status(acp::ToolCallStatus::InProgress),
        );
        update.meta = Some(subagent_meta("agent-1"));
        turn.tool_call_update(update, &mut emit);
        assert_eq!(turn.items.len(), 1);
        assert_eq!(
            progress_of(&turn).as_deref(),
            Some("1 tool call · now: Read src/auth.rs")
        );
    }

    #[test]
    fn subagent_call_needing_permission_becomes_an_item() {
        let mut turn = builder();
        let mut emit = Emit::new();
        turn.tool_call(agent_call(), &mut emit);
        turn.tool_call(child_call("c1", "Read src/auth.rs"), &mut emit);
        let update = acp::ToolCallUpdate::new("c1", acp::ToolCallUpdateFields::new());
        let (item_id, _) = turn.tool_item(&update, &mut emit);
        assert_eq!(item_id, "t1-2");
        assert_eq!(turn.items.len(), 2);
        assert_eq!(progress_of(&turn), None);
    }
}
