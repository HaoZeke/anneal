//! Every FILE:LINE reference in the org docs points at the item its sentence
//! names. The names are the =verbatim= and ~code~ spans written before the
//! reference in its clause, back to the previous reference, a `;` or the start
//! of the paragraph. Each cited line or range holds one of them as a whole
//! word, or is module notes when one of them is the file's stem.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

const EXTENSIONS: [&str; 5] = ["rs", "py", "capnp", "md", "toml"];
const BEFORE_SPAN: &str = "-('\"{";
const AFTER_SPAN: &str = "-.,:!?;'\")}\\[";

struct Citation {
    line: usize,
    file: String,
    spec: String,
    names: Vec<String>,
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("{}: {error}", dir.display()))
        .map(|entry| entry.expect("directory entry").path())
        .collect();
    entries.sort();
    for path in entries {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if !path.is_dir() {
            out.push(path);
        } else if !name.starts_with('.') && name != "target" && name != "vendor" {
            walk(&path, out);
        }
    }
}

fn relative(root: &Path, path: &Path) -> String {
    let parts: Vec<_> = path
        .strip_prefix(root)
        .expect("path under the root")
        .iter()
        .map(|part| part.to_string_lossy())
        .collect();
    parts.join("/")
}

fn run_end(text: &[char], from: usize, keep: impl Fn(char) -> bool) -> usize {
    text[from..]
        .iter()
        .position(|&c| !keep(c))
        .map_or(text.len(), |offset| from + offset)
}

fn is_word(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn is_path(c: char) -> bool {
    is_word(c) || c == '-'
}

/// Headings, keyword lines and list items start a paragraph.
fn starts_paragraph(line: &str) -> bool {
    let rest = line.trim_start();
    let spaced = |at: usize| rest[at..].starts_with(char::is_whitespace);
    let stars = rest.len() - rest.trim_start_matches('*').len();
    let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    (stars > 0 && spaced(stars))
        || rest.starts_with('#')
        || (rest.starts_with(['-', '+']) && spaced(1))
        || (digits > 0 && rest[digits..].starts_with(['.', ')']) && spaced(digits + 1))
}

/// The first line number and the lines of each paragraph.
fn paragraphs(text: &str) -> Vec<(usize, Vec<&str>)> {
    let mut out = Vec::new();
    let mut current = Vec::new();
    let mut first = 0;
    for (index, line) in text.split('\n').chain([""]).enumerate() {
        let blank = line.trim().is_empty();
        if (blank || starts_paragraph(line)) && !current.is_empty() {
            out.push((first, std::mem::take(&mut current)));
        }
        if blank {
            continue;
        }
        if current.is_empty() {
            first = index + 1;
        }
        current.push(line);
    }
    out
}

/// Blanks LaTeX fragments and link targets, in which Org renders no spans.
fn mask(text: &[char]) -> Vec<char> {
    let mut out = text.to_vec();
    let mut at = 0;
    while at < text.len() {
        let close: &[char] = match (text[at], text.get(at + 1)) {
            ('\\', Some('(')) => &['\\', ')'],
            ('\\', Some('[')) => &['\\', ']'],
            ('[', Some('[')) => &[']'],
            _ => &[],
        };
        let end = if close.is_empty() {
            None
        } else {
            (at + 2..text.len()).find(|&k| text[k..].starts_with(close))
        };
        match end {
            Some(end) => {
                let end = end + close.len();
                out[at..end].fill(' ');
                at = end;
            }
            None => at += 1,
        }
    }
    out
}

/// The start, end and body of each =verbatim= or ~code~ span under Org's
/// emphasis rules, the end past the closing marker.
fn spans(text: &[char]) -> Vec<(usize, usize, String)> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < text.len() {
        let marker = text[at];
        let opens = (marker == '=' || marker == '~')
            && (at == 0 || text[at - 1].is_whitespace() || BEFORE_SPAN.contains(text[at - 1]))
            && text.get(at + 1).is_some_and(|c| !c.is_whitespace());
        let close = if opens {
            (at + 2..text.len()).find(|&k| {
                text[k] == marker
                    && !text[k - 1].is_whitespace()
                    && text
                        .get(k + 1)
                        .is_none_or(|&c| c.is_whitespace() || AFTER_SPAN.contains(c))
            })
        } else {
            None
        };
        match close {
            Some(close) => {
                out.push((at, close + 1, text[at + 1..close].iter().collect()));
                at = close + 1;
            }
            None => at += 1,
        }
    }
    out
}

/// The file and line spec of a FILE:SPEC reference starting at `start`, and
/// where it ends.
fn citation_at(text: &[char], start: usize) -> Option<(String, String, usize)> {
    if start > 0 && (is_path(text[start - 1]) || text[start - 1] == '.' || text[start - 1] == '/') {
        return None;
    }
    let mut at = start;
    let stem_end = loop {
        let end = run_end(text, at, is_path);
        if end == at {
            return None;
        }
        match text.get(end) {
            Some('/') => at = end + 1,
            Some('.') => break end + 1,
            _ => return None,
        }
    };
    let colon = EXTENSIONS.iter().find_map(|extension| {
        let extension: Vec<char> = extension.chars().collect();
        let colon = stem_end + extension.len();
        (text[stem_end..].starts_with(&extension) && text.get(colon) == Some(&':')).then_some(colon)
    })?;
    let number =
        |from: usize| Some(run_end(text, from, |c| c.is_ascii_digit())).filter(|&end| end > from);
    let range = |from: usize| {
        number(from).map(|end| match text.get(end) {
            Some('-') => number(end + 1).unwrap_or(end),
            _ => end,
        })
    };
    let mut end = range(colon + 1)?;
    while let Some(next) = match text.get(end) {
        Some('/') => Some(end + 1),
        Some(',') if text.get(end + 1) == Some(&' ') => Some(end + 2),
        Some(',') => Some(end + 1),
        _ => None,
    }
    .and_then(range)
    {
        end = next;
    }
    let file = text[start..colon].iter().collect();
    let spec = text[colon + 1..end].iter().collect();
    Some((file, spec, end))
}

/// The item a span names: the last segment of its leading path, unless the
/// path goes on as a file path.
fn item_name(span: &[char]) -> Option<String> {
    let span = span
        .strip_prefix(&['#', '!', '['])
        .or_else(|| span.strip_prefix(&['#', '[']))
        .unwrap_or(span);
    let identifier = |from: usize| {
        span.get(from)
            .is_some_and(|&c| c.is_ascii_alphabetic() || c == '_')
            .then(|| run_end(span, from + 1, is_word))
    };
    let mut start = if span.starts_with(&[':', ':']) { 2 } else { 0 };
    let mut end = identifier(start)?;
    loop {
        let separator = if span[end..].starts_with(&[':', ':']) {
            2
        } else if span.get(end) == Some(&'.') {
            1
        } else {
            break;
        };
        let Some(next) = identifier(end + separator) else {
            break;
        };
        start = end + separator;
        end = next;
    }
    (span.get(end) != Some(&'/')).then(|| span[start..end].iter().collect())
}

/// Org renders =a=/=b= as one span; each part names its own item.
fn item_names(body: &str) -> Vec<String> {
    let body: Vec<char> = body.chars().collect();
    let marker = |at: usize| body.get(at).is_some_and(|&c| c == '=' || c == '~');
    let mut parts = Vec::new();
    let mut from = 0;
    let mut at = 0;
    while at < body.len() {
        if marker(at) && body.get(at + 1) == Some(&'/') {
            let next = run_end(&body, at + 2, char::is_whitespace);
            if marker(next) {
                parts.push(&body[from..at]);
                from = next + 1;
                at = from;
                continue;
            }
        }
        at += 1;
    }
    parts.push(&body[from..]);
    parts.into_iter().filter_map(item_name).collect()
}

fn citations(text: &str) -> Vec<Citation> {
    let mut out = Vec::new();
    for (first, lines) in paragraphs(text) {
        let mut joined: Vec<char> = Vec::new();
        let mut starts = Vec::new();
        for (offset, line) in lines.iter().enumerate() {
            if !joined.is_empty() {
                joined.push(' ');
            }
            starts.push((joined.len(), first + offset));
            joined.extend(line.trim().chars());
        }
        let masked = mask(&joined);
        let found = spans(&masked);
        let mut previous = 0;
        let mut at = 0;
        while at < joined.len() {
            let Some((file, spec, end)) = citation_at(&joined, at) else {
                at += 1;
                continue;
            };
            let clause = (previous..at)
                .rev()
                .find(|&k| masked[k] == ';' && !found.iter().any(|&(a, b, _)| a <= k && k < b))
                .map_or(previous, |k| k + 1);
            let names = found
                .iter()
                .filter(|&&(a, b, _)| a >= clause && b <= at)
                .flat_map(|(_, _, body)| item_names(body))
                .collect();
            let line = starts
                .iter()
                .rev()
                .find(|&&(offset, _)| offset <= at)
                .map_or(first, |&(_, line)| line);
            out.push(Citation {
                line,
                file,
                spec,
                names,
            });
            previous = end;
            at = end;
        }
    }
    out
}

fn ranges(spec: &str) -> Vec<(usize, usize)> {
    spec.split(['/', ','])
        .map(|part| {
            let part = part.trim();
            let (first, last) = part.split_once('-').unwrap_or((part, part));
            let number = |text: &str| -> usize { text.parse().expect("line number") };
            (number(first), number(last))
        })
        .collect()
}

fn holds_word(line: &str, word: &str) -> bool {
    line.match_indices(word).any(|(at, _)| {
        !line[..at].chars().next_back().is_some_and(is_word)
            && !line[at + word.len()..].chars().next().is_some_and(is_word)
    })
}

fn check(
    root: &Path,
    sources: &[String],
    cache: &mut HashMap<String, String>,
    citation: &Citation,
) -> Result<(), String> {
    let path = if root.join(&citation.file).is_file() {
        citation.file.clone()
    } else {
        let suffix = format!("/{}", citation.file);
        let ends: Vec<&String> = sources
            .iter()
            .filter(|path| path.ends_with(&suffix))
            .collect();
        let under_src: Vec<&String> = ends
            .iter()
            .copied()
            .filter(|path| path.starts_with("src/"))
            .collect();
        let candidates = if under_src.is_empty() {
            ends
        } else {
            under_src
        };
        match candidates.as_slice() {
            [path] => (*path).clone(),
            [] => return Err("names no file in the tree".into()),
            many => return Err(format!("names {} files", many.len())),
        }
    };
    if citation.names.is_empty() {
        return Err("names no item before it".into());
    }
    let source = cache.entry(path.clone()).or_insert_with(|| {
        fs::read_to_string(root.join(&path)).unwrap_or_else(|error| panic!("{path}: {error}"))
    });
    let lines: Vec<&str> = source.split('\n').collect();
    let stem = Path::new(&path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("");
    let names_module = citation.names.iter().any(|name| name == stem);
    let missing: Vec<String> = ranges(&citation.spec)
        .into_iter()
        .filter(|&(first, last)| {
            let cited = if 1 <= first && first <= last && last <= lines.len() {
                &lines[first - 1..last]
            } else {
                &[][..]
            };
            let notes = names_module
                && !cited.is_empty()
                && cited
                    .iter()
                    .all(|line| line.trim_start().starts_with("//!"));
            !notes
                && !citation
                    .names
                    .iter()
                    .any(|name| cited.iter().any(|line| holds_word(line, name)))
        })
        .map(|(first, last)| {
            if first == last {
                first.to_string()
            } else {
                format!("{first}-{last}")
            }
        })
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} holds none of {}",
            missing.join(", "),
            citation.names.join(", ")
        ))
    }
}

#[test]
fn every_org_line_reference_holds_the_item_it_names() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    walk(root, &mut files);
    let sources: Vec<String> = files
        .iter()
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| EXTENSIONS.contains(&extension))
        })
        .map(|path| relative(root, path))
        .collect();
    let orgmode = root.join("docs/orgmode");
    let mut cache = HashMap::new();
    let mut checked = 0;
    let mut failures = Vec::new();
    for doc in files
        .iter()
        .filter(|path| path.starts_with(&orgmode) && path.extension().is_some_and(|e| e == "org"))
    {
        let text =
            fs::read_to_string(doc).unwrap_or_else(|error| panic!("{}: {error}", doc.display()));
        for citation in citations(&text) {
            checked += 1;
            if let Err(reason) = check(root, &sources, &mut cache, &citation) {
                failures.push(format!(
                    "{}:{}: {}:{} {reason}",
                    relative(root, doc),
                    citation.line,
                    citation.file,
                    citation.spec
                ));
            }
        }
    }
    assert!(checked > 0, "no line references found under docs/orgmode");
    assert!(
        failures.is_empty(),
        "{} of {checked} line references under docs/orgmode do not hold the items they name:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
