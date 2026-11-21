//! acpc's directories and the private `CODEX_HOME` given to the Codex TUI.

use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use toml_edit::DocumentMut;
use toml_edit::Item;
use toml_edit::Table;

/// `$ACPC_CONFIG`, else `$XDG_CONFIG_HOME/acpc/config.toml`.
pub fn config_path() -> PathBuf {
    if let Some(path) = std::env::var_os("ACPC_CONFIG") {
        return PathBuf::from(path);
    }
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("acpc")
        .join("config.toml")
}

/// `$ACPC_HOME`, else `$XDG_DATA_HOME/acpc`: logs, sockets, the TUI's home.
pub fn data_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("ACPC_HOME") {
        return PathBuf::from(path);
    }
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("acpc")
}

fn create_private_dir(path: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(path).with_context(|| format!("creating {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
            .with_context(|| format!("securing {}", path.display()))?;
    }
    Ok(())
}

/// First line of the TUI's config.toml. Older acpc versions wrote a different
/// line starting with the same prefix; both are replaced on launch.
const MANAGED_HEADER_PREFIX: &str = "# Managed by acpc";
const MANAGED_HEADER: &str = "# Managed by acpc: the settings acpc needs are reset on every launch; \
settings saved from the TUI (e.g. /statusline, /theme) are kept.\n";

/// Settings acpc forces on the embedded Codex TUI. The TUI never talks to
/// OpenAI through acpc, so every Codex network feature is switched off.
/// Everything else in the file belongs to the TUI and survives restarts.
const MANAGED_CODEX_CONFIG: &str = r#"check_for_update_on_startup = false

[analytics]
enabled = false

[feedback]
enabled = false

[otel]
exporter = "none"
trace_exporter = "none"
metrics_exporter = "none"

# Codex's welcome art, tips and server-version notices describe Codex, not
# the ACP agent.
[tui]
show_tooltips = false
show_server_version_notice = false

[tui.effects]
welcome = false
"#;

pub struct Home {
    pub codex_home: PathBuf,
    pub log_dir: PathBuf,
    pub run_dir: PathBuf,
}

impl Home {
    pub fn prepare() -> anyhow::Result<Self> {
        let data_dir = data_dir();
        let codex_home = data_dir.join("codex");
        let log_dir = data_dir.join("log");
        let run_dir = data_dir.join("run");
        for dir in [&data_dir, &codex_home, &log_dir, &run_dir] {
            create_private_dir(dir)?;
        }
        let config = codex_home.join("config.toml");
        let existing = match std::fs::read_to_string(&config) {
            Ok(existing) => existing,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(err) => {
                return Err(err).with_context(|| format!("reading {}", config.display()));
            }
        };
        std::fs::write(&config, merged_codex_config(&existing)?)
            .with_context(|| format!("writing {}", config.display()))?;
        Ok(Self {
            codex_home,
            log_dir,
            run_dir,
        })
    }

    pub fn socket_path(&self) -> PathBuf {
        self.run_dir
            .join(format!("bridge-{}.sock", std::process::id()))
    }
}

/// The TUI's config.toml: `existing` with acpc's managed settings enforced on
/// top. Settings the TUI saved itself (status line, theme, ...) are kept.
fn merged_codex_config(existing: &str) -> anyhow::Result<String> {
    let mut body = existing;
    while body.starts_with(MANAGED_HEADER_PREFIX) {
        body = body.split_once('\n').map_or("", |(_, rest)| rest);
    }
    let mut doc = body.parse::<DocumentMut>().unwrap_or_else(|err| {
        tracing::warn!(error = %err, "resetting unreadable Codex TUI config.toml");
        DocumentMut::new()
    });
    let managed = MANAGED_CODEX_CONFIG
        .parse::<DocumentMut>()
        .context("parsing acpc's managed Codex config")?;
    enforce(doc.as_table_mut(), managed.as_table());
    Ok(format!("{MANAGED_HEADER}{doc}"))
}

/// Sets every key of `managed` in `target`, descending into tables both share.
fn enforce(target: &mut Table, managed: &Table) {
    for (key, item) in managed.iter() {
        match (target.get_mut(key), item) {
            (Some(Item::Table(existing)), Item::Table(table)) => enforce(existing, table),
            _ => {
                target.insert(key, item.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn parse(text: &str) -> toml::Value {
        text.parse::<toml::Table>().expect("valid TOML").into()
    }

    #[test]
    fn fresh_config_holds_the_managed_settings() {
        let merged = merged_codex_config("").expect("merged");
        assert!(merged.starts_with(MANAGED_HEADER));
        assert_eq!(parse(&merged), parse(MANAGED_CODEX_CONFIG));
    }

    #[test]
    fn keeps_settings_saved_from_the_tui() {
        let existing = r#"# Managed by acpc; rewritten on every launch. Edit ~/.config/acpc/config.toml instead.
check_for_update_on_startup = false

[analytics]
enabled = false

[tui]
show_tooltips = false
screen_reader_detection_done = true
status_line = ["model-with-reasoning", "context-remaining", "git-branch"]
status_line_use_colors = true
theme = "dracula"
"#;
        let merged = merged_codex_config(existing).expect("merged");
        assert_eq!(merged.matches(MANAGED_HEADER_PREFIX).count(), 1);
        let value = parse(&merged);
        let tui = &value["tui"];
        assert_eq!(
            tui["status_line"],
            toml::Value::Array(vec![
                "model-with-reasoning".into(),
                "context-remaining".into(),
                "git-branch".into(),
            ])
        );
        assert_eq!(tui["status_line_use_colors"], toml::Value::Boolean(true));
        assert_eq!(
            tui["screen_reader_detection_done"],
            toml::Value::Boolean(true)
        );
        assert_eq!(tui["theme"], toml::Value::String("dracula".into()));
        assert_eq!(
            tui["show_server_version_notice"],
            toml::Value::Boolean(false)
        );
        assert_eq!(tui["effects"]["welcome"], toml::Value::Boolean(false));
        assert_eq!(
            value["otel"]["exporter"],
            toml::Value::String("none".into())
        );
    }

    #[test]
    fn managed_settings_win_over_edits() {
        let existing = r#"check_for_update_on_startup = true
analytics = { enabled = true }

[tui]
show_tooltips = true
"#;
        let value = parse(&merged_codex_config(existing).expect("merged"));
        assert_eq!(
            value["check_for_update_on_startup"],
            toml::Value::Boolean(false)
        );
        assert_eq!(value["analytics"]["enabled"], toml::Value::Boolean(false));
        assert_eq!(value["tui"]["show_tooltips"], toml::Value::Boolean(false));
    }

    #[test]
    fn unreadable_config_is_reset() {
        let merged = merged_codex_config("[tui\nstatus_line = ").expect("merged");
        assert_eq!(parse(&merged), parse(MANAGED_CODEX_CONFIG));
    }

    #[test]
    fn merging_is_stable_across_launches() {
        let once = merged_codex_config("[tui]\nstatus_line = [\"git-branch\"]\n").expect("merged");
        assert_eq!(merged_codex_config(&once).expect("merged"), once);
    }
}
