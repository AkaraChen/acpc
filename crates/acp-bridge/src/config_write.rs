//! The TUI saves some preferences through `config/batchWrite` and
//! `config/value/write`. acpc keeps its settings in the agent session, so most
//! of those writes are dropped. The default model and reasoning effort are
//! the exception: `/model` offers them as the default for later sessions, and
//! the TUI reads them back from its `config.toml` when it starts.

use std::path::Path;

use codex_app_server_protocol::ConfigEdit;
use codex_protocol::openai_models::ReasoningEffort;
use serde_json::Value;
use toml_edit::DocumentMut;
use toml_edit::Item;

use crate::threads::TurnSettings;

/// Keys of the TUI's config.toml that the bridge saves.
const SAVED_KEYS: &[&str] = &["model", "model_reasoning_effort"];

/// Apply the saved-key edits to the config file at `path`. A null value
/// removes the key. Returns whether the file changed.
pub(crate) fn save_edits(path: &Path, edits: &[ConfigEdit]) -> std::io::Result<bool> {
    let edits: Vec<&ConfigEdit> = edits
        .iter()
        .filter(|edit| SAVED_KEYS.contains(&edit.key_path.as_str()))
        .collect();
    if edits.is_empty() {
        return Ok(false);
    }
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(err) => return Err(err),
    };
    let mut doc = text.parse::<DocumentMut>().unwrap_or_default();
    let mut changed = false;
    for edit in edits {
        let key = edit.key_path.as_str();
        match &edit.value {
            Value::String(value) => {
                let unchanged = doc.get(key).and_then(Item::as_str) == Some(value.as_str());
                if !unchanged {
                    doc[key] = toml_edit::value(value.as_str());
                    changed = true;
                }
            }
            Value::Null => changed |= doc.remove(key).is_some(),
            // The saved keys are strings; anything else is not ours to write.
            _ => {}
        }
    }
    if changed {
        std::fs::write(path, doc.to_string())?;
    }
    Ok(changed)
}

/// The default model and effort saved by `/model`, for a new session to start with.
pub(crate) fn saved_defaults(path: &Path) -> TurnSettings {
    let Some(doc) = std::fs::read_to_string(path)
        .ok()
        .and_then(|text| text.parse::<DocumentMut>().ok())
    else {
        return TurnSettings::default();
    };
    let text = |key: &str| doc.get(key).and_then(Item::as_str).map(str::to_string);
    TurnSettings {
        model: text("model").filter(|model| !model.is_empty()),
        effort: text("model_reasoning_effort").and_then(|effort| {
            serde_json::from_value::<ReasoningEffort>(Value::String(effort)).ok()
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_app_server_protocol::MergeStrategy;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    fn edit(key: &str, value: Value) -> ConfigEdit {
        ConfigEdit {
            key_path: key.to_string(),
            value,
            merge_strategy: MergeStrategy::Replace,
        }
    }

    #[test]
    fn saves_model_and_effort_and_keeps_the_rest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[tui]\nstatus_line = [\"git-branch\"]\n").expect("write");
        let changed = save_edits(
            &path,
            &[
                edit("model", json!("opus")),
                edit("model_reasoning_effort", json!("high")),
                edit("tui.theme", json!("dracula")),
            ],
        )
        .expect("save");
        assert!(changed);
        let saved = std::fs::read_to_string(&path).expect("read");
        let doc = saved.parse::<DocumentMut>().expect("toml");
        assert_eq!(doc["model"].as_str(), Some("opus"));
        assert_eq!(doc["model_reasoning_effort"].as_str(), Some("high"));
        assert!(saved.contains("status_line"));
        assert!(doc.get("tui").and_then(|tui| tui.get("theme")).is_none());
    }

    #[test]
    fn saved_defaults_read_back() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        assert!(saved_defaults(&path).model.is_none());
        save_edits(
            &path,
            &[
                edit("model", json!("opus")),
                edit("model_reasoning_effort", json!("high")),
            ],
        )
        .expect("save");
        let defaults = saved_defaults(&path);
        assert_eq!(defaults.model.as_deref(), Some("opus"));
        assert_eq!(defaults.effort, Some(ReasoningEffort::High));
    }

    #[test]
    fn null_removes_and_repeats_change_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        assert!(!save_edits(&path, &[edit("tui.theme", json!("x"))]).expect("save"));
        assert!(!path.exists());
        assert!(save_edits(&path, &[edit("model", json!("opus"))]).expect("save"));
        assert!(!save_edits(&path, &[edit("model", json!("opus"))]).expect("save"));
        assert!(save_edits(&path, &[edit("model", Value::Null)]).expect("save"));
        let saved = std::fs::read_to_string(&path).expect("read");
        assert!(!saved.contains("model"));
    }
}
