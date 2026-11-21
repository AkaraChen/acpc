//! Conversions between Codex user input and ACP content blocks.

use std::path::Path;

use acp_client::AgentCaps;
use acp_client::schema as acp;
use base64::Engine;
use codex_app_server_protocol::ImageReference;
use codex_app_server_protocol::UserInput;
use codex_utils_image::EncodedImage;
use codex_utils_image::PromptImageMode;

/// Codex core's default resize policy for prompt images (longest side at most
/// 2048 px). The TUI leaves resizing to core, which acpc does not run, so the
/// bridge applies it before an image goes to the agent.
const PROMPT_IMAGE_MODE: PromptImageMode = PromptImageMode::HIGH_DETAIL;

/// Translate a Codex turn input into an ACP prompt, honouring the agent's
/// prompt capabilities. Unsupported attachments become text notes rather
/// than being sent against the spec.
pub(crate) fn prompt_blocks(
    input: &[UserInput],
    caps: &AgentCaps,
    cwd: &Path,
) -> Vec<acp::ContentBlock> {
    let mut blocks = Vec::new();
    for item in input {
        match item {
            UserInput::Text { text, .. } => {
                if !text.is_empty() {
                    blocks.push(acp::ContentBlock::Text(acp::TextContent::new(text.clone())));
                }
            }
            UserInput::LocalImage { path, .. } => {
                let path = absolute(cwd, path);
                match image_block(&path, caps) {
                    Some(block) => blocks.push(block),
                    None => blocks.push(link_block(&path)),
                }
            }
            UserInput::Image { image, .. } => blocks.push(inline_image_block(image, caps)),
            UserInput::Mention { path, .. } => {
                blocks.push(link_block(&absolute(cwd, Path::new(path))));
            }
            UserInput::Skill { path, .. } => {
                blocks.push(link_block(&absolute(cwd, path)));
            }
            UserInput::LocalAudio { path } => {
                blocks.push(link_block(&absolute(cwd, path)));
            }
            UserInput::Audio { .. } => {}
        }
    }
    blocks
}

/// An image the TUI sends inline (pasted or attached images arrive as `data:`
/// URLs). It must reach the agent as an image block: agents turn other blocks
/// into text, and a base64 payload as text floods the model's context.
fn inline_image_block(image: &ImageReference, caps: &AgentCaps) -> acp::ContentBlock {
    let url = match image {
        ImageReference::Inline { url } => url,
        ImageReference::File { .. } => {
            return note("[image omitted: file references are not supported]");
        }
    };
    let is_data_url = url
        .get(..5)
        .is_some_and(|scheme| scheme.eq_ignore_ascii_case("data:"));
    if !is_data_url {
        // A remote URL is short, so a link is safe.
        return acp::ContentBlock::ResourceLink(acp::ResourceLink::new("image", url.clone()));
    }
    if !caps.prompt_image() {
        return note("[image omitted: this agent does not accept images]");
    }
    match codex_utils_image::load_data_url_for_prompt(
        url,
        PROMPT_IMAGE_MODE,
        codex_utils_image::PromptImageDecoder::default(),
    ) {
        Ok(encoded) => acp::ContentBlock::Image(image_content(&encoded)),
        Err(err) => {
            tracing::warn!("replacing unreadable pasted image with a note: {err}");
            note(&format!("[image omitted: {err}]"))
        }
    }
}

fn image_content(encoded: &EncodedImage) -> acp::ImageContent {
    let data = base64::engine::general_purpose::STANDARD.encode(&encoded.bytes);
    acp::ImageContent::new(data, encoded.mime.clone())
}

fn note(text: &str) -> acp::ContentBlock {
    acp::ContentBlock::Text(acp::TextContent::new(text.to_string()))
}

fn absolute(cwd: &Path, path: &Path) -> std::path::PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

fn link_block(path: &Path) -> acp::ContentBlock {
    let name = path.file_name().map_or_else(
        || path.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    );
    acp::ContentBlock::ResourceLink(acp::ResourceLink::new(name, file_uri(path)))
}

pub(crate) fn file_uri(path: &Path) -> String {
    format!("file://{}", path.display())
}

fn image_block(path: &Path, caps: &AgentCaps) -> Option<acp::ContentBlock> {
    if !caps.prompt_image() {
        return None;
    }
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase);
    if !matches!(
        extension.as_deref(),
        Some("png" | "jpg" | "jpeg" | "gif" | "webp")
    ) {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    let loaded = codex_utils_image::load_for_prompt_bytes(path, bytes, PROMPT_IMAGE_MODE);
    let encoded = match loaded {
        Ok(encoded) => encoded,
        Err(err) => {
            tracing::warn!("sending {} as a link: {err}", path.display());
            return None;
        }
    };
    Some(acp::ContentBlock::Image(
        image_content(&encoded).uri(file_uri(path)),
    ))
}

/// Text to display for an ACP content block.
pub(crate) fn display_text(block: &acp::ContentBlock) -> String {
    match block {
        acp::ContentBlock::Text(text) => text.text.clone(),
        acp::ContentBlock::ResourceLink(link) => {
            let label = link.title.as_deref().unwrap_or(&link.name);
            format!("[{label}]({})", link.uri)
        }
        acp::ContentBlock::Resource(resource) => match &resource.resource {
            acp::EmbeddedResourceResource::TextResourceContents(text) => text.text.clone(),
            acp::EmbeddedResourceResource::BlobResourceContents(blob) => {
                format!("[resource]({})", blob.uri)
            }
            _ => String::new(),
        },
        acp::ContentBlock::Image(image) => match &image.uri {
            Some(uri) => format!("[image]({uri})"),
            None => "[image]".to_string(),
        },
        acp::ContentBlock::Audio(_) => "[audio]".to_string(),
        _ => String::new(),
    }
}

/// A user message as Codex input, for replayed history.
pub(crate) fn user_input(block: &acp::ContentBlock) -> Option<UserInput> {
    if crate::handoff::is_handoff(block) {
        return None;
    }
    match block {
        acp::ContentBlock::Text(text) => {
            crate::handoff::strip(&text.text).map(|text| UserInput::Text {
                text: text.to_string(),
                text_elements: Vec::new(),
            })
        }
        acp::ContentBlock::ResourceLink(link) => {
            let path = link.uri.strip_prefix("file://").unwrap_or(&link.uri);
            Some(UserInput::Mention {
                name: link.name.clone(),
                path: path.to_string(),
            })
        }
        acp::ContentBlock::Image(image) => image
            .uri
            .as_deref()
            .and_then(|uri| uri.strip_prefix("file://"))
            .map(|path| UserInput::LocalImage {
                detail: None,
                path: path.into(),
            }),
        other => {
            let text = display_text(other);
            (!text.is_empty()).then_some(UserInput::Text {
                text,
                text_elements: Vec::new(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn caps(image: bool) -> AgentCaps {
        let init = serde_json::json!({
            "protocolVersion": 1,
            "agentCapabilities": {"promptCapabilities": {"image": image}},
        });
        AgentCaps {
            init: serde_json::from_value(init).expect("initialize response"),
        }
    }

    fn png_bytes(width: u32, height: u32) -> Vec<u8> {
        let image = image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x ^ y) % 256) as u8])
        });
        let mut bytes = std::io::Cursor::new(Vec::new());
        image
            .write_to(&mut bytes, image::ImageFormat::Png)
            .expect("encode png");
        bytes.into_inner()
    }

    fn pasted(bytes: &[u8]) -> UserInput {
        UserInput::Image {
            image: ImageReference::Inline {
                url: codex_utils_image::data_url_from_bytes("image/png", bytes),
            },
            detail: None,
        }
    }

    fn only_block(input: UserInput, caps: &AgentCaps) -> acp::ContentBlock {
        let mut blocks = prompt_blocks(&[input], caps, Path::new("/"));
        assert_eq!(blocks.len(), 1);
        blocks.remove(0)
    }

    fn decoded_size(image: &acp::ImageContent) -> (u32, u32) {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&image.data)
            .expect("base64");
        let decoded = image::load_from_memory(&bytes).expect("decode");
        (decoded.width(), decoded.height())
    }

    #[test]
    fn pasted_image_becomes_a_resized_image_block() {
        let block = only_block(pasted(&png_bytes(4096, 64)), &caps(true));
        let acp::ContentBlock::Image(image) = block else {
            panic!("expected an image block, got {block:?}");
        };
        assert_eq!(image.mime_type, "image/png");
        assert_eq!(decoded_size(&image), (2048, 32));
    }

    #[test]
    fn small_pasted_image_keeps_its_size() {
        let block = only_block(pasted(&png_bytes(40, 30)), &caps(true));
        let acp::ContentBlock::Image(image) = block else {
            panic!("expected an image block, got {block:?}");
        };
        assert_eq!(decoded_size(&image), (40, 30));
    }

    #[test]
    fn pasted_image_is_never_sent_as_text_to_an_agent_without_images() {
        let block = only_block(pasted(&png_bytes(40, 30)), &caps(false));
        assert_eq!(
            display_text(&block),
            "[image omitted: this agent does not accept images]"
        );
    }

    #[test]
    fn unreadable_pasted_image_becomes_a_short_note() {
        let input = UserInput::Image {
            image: ImageReference::Inline {
                url: "data:image/png;base64,bm90IGFuIGltYWdl".to_string(),
            },
            detail: None,
        };
        let block = only_block(input, &caps(true));
        let acp::ContentBlock::Text(text) = &block else {
            panic!("expected a text note, got {block:?}");
        };
        assert!(text.text.starts_with("[image omitted:"), "{}", text.text);
        assert!(!text.text.contains("bm90IGFuIGltYWdl"), "{}", text.text);
    }

    #[test]
    fn remote_image_url_stays_a_link() {
        let input = UserInput::Image {
            image: ImageReference::Inline {
                url: "https://example.com/cat.png".to_string(),
            },
            detail: None,
        };
        let block = only_block(input, &caps(true));
        assert_eq!(display_text(&block), "[image](https://example.com/cat.png)");
    }

    #[test]
    fn local_image_is_resized_too() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("shot.png");
        std::fs::write(&path, png_bytes(64, 4096)).expect("write");
        let input = UserInput::LocalImage {
            detail: None,
            path: path.clone(),
        };
        let block = only_block(input, &caps(true));
        let acp::ContentBlock::Image(image) = block else {
            panic!("expected an image block, got {block:?}");
        };
        assert_eq!(image.uri.as_deref(), Some(file_uri(&path).as_str()));
        assert_eq!(decoded_size(&image), (32, 2048));
    }
}
