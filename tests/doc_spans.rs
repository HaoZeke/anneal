//! Every =verbatim= and ~code~ span in the org docs ends where it is written.
//! Org opens such a span at a marker after a space or one of -('"{ and before
//! a non-space, and ends it at the first marker after a non-space that comes
//! before a space, a quote or punctuation, so =history="shared"= ends after
//! history and =cap >= 50000= after the greater-than sign, while
//! =worst_bound=/=worst_bound_scaled= stays one span, since Org ends no span
//! before a slash and opens none after one. A span cut short or never opened
//! shows in one of two ways: a span runs straight into a quoted word, as
//! history runs into "shared", or a marker placed like a closing one, after a
//! letter, a digit, a closing bracket or a quote and before a space or
//! punctuation, closes no span, as the last equals sign of =cap >= 50000=
//! does. Spans run together show as a span that holds its own marker after a
//! non-space and before a separator and then, past any spaces, the marker
//! again before a non-space. A marker inside a span that comes before a letter
//! or digit, as in =batch_size=1=, does not end it.

use std::fs;
use std::path::{Path, PathBuf};

const BEFORE_SPAN: &str = "-('\"{";
const AFTER_SPAN: &str = "-.,:!?;'\")}\\[";
const BEFORE_CLOSER: &str = ")]}\"'";
const AFTER_CLOSER: &str = ".,:;!?)}\"'";
const LITERAL_BLOCKS: [&str; 4] = ["src", "example", "export", "comment"];

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("{}: {error}", dir.display()))
        .map(|entry| entry.expect("directory entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().is_some_and(|extension| extension == "org") {
            out.push(path);
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

/// List items start a paragraph.
fn starts_item(line: &str) -> bool {
    let rest = line.trim_start();
    let spaced = |at: usize| rest[at..].starts_with(char::is_whitespace);
    let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    (rest.starts_with(['-', '+']) && spaced(1))
        || (digits > 0 && rest[digits..].starts_with(['.', ')']) && spaced(digits + 1))
}

/// The first line number and the text of each paragraph, heading and table
/// cell, inside which Org pairs span markers. Source, example, export and
/// comment blocks, fixed-width lines, comments and keyword lines hold none.
fn units(text: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut paragraph: Option<(usize, String)> = None;
    let mut block: Option<String> = None;
    for (index, line) in text.split('\n').enumerate() {
        let rest = line.trim();
        let lower = rest.to_ascii_lowercase();
        let literal = lower
            .strip_prefix("#+begin_")
            .map(|name| name.split_whitespace().next().unwrap_or(""))
            .filter(|name| LITERAL_BLOCKS.contains(name));
        let skipped = rest == "#"
            || rest.starts_with("# ")
            || rest.starts_with("#+")
            || rest == ":"
            || rest.starts_with(": ");
        let heading =
            line.starts_with('*') && line.trim_start_matches('*').starts_with([' ', '\t']);
        let row = rest.starts_with('|');
        if block.is_some() || skipped || rest.is_empty() || heading || row || starts_item(line) {
            out.extend(paragraph.take());
        }
        if let Some(end) = &block {
            if lower.starts_with(end.as_str()) {
                block = None;
            }
        } else if let Some(name) = literal {
            block = Some(format!("#+end_{name}"));
        } else if heading {
            out.push((index + 1, line.to_string()));
        } else if row {
            let cells = rest.trim_matches('|').split('|');
            out.extend(cells.map(|cell| (index + 1, cell.to_string())));
        } else if !skipped && !rest.is_empty() {
            let (_, body) = paragraph.get_or_insert_with(|| (index + 1, String::new()));
            if !body.is_empty() {
                body.push('\n');
            }
            body.push_str(line);
        }
    }
    out.extend(paragraph);
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

/// The start and end of each =verbatim= or ~code~ span under Org's emphasis
/// rules, the end past the closing marker.
fn spans(text: &[char]) -> Vec<(usize, usize)> {
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
                out.push((at, close + 1));
                at = close + 1;
            }
            None => at += 1,
        }
    }
    out
}

/// How often `body`, the inside of a span, holds the end of one span and the
/// start of the next: its marker after a non-space and before a run of
/// characters other than letters, digits, spaces and the marker, then past any
/// spaces the marker again before a non-space, as =a=/=b= does.
fn joins(body: &[char], marker: char) -> usize {
    (1..body.len())
        .filter(|&at| body[at] == marker && !body[at - 1].is_whitespace())
        .filter(|&at| {
            let separator = run_end(body, at + 1, |c| {
                !is_word(c) && !c.is_whitespace() && c != marker
            });
            let next = run_end(body, separator, char::is_whitespace);
            separator > at + 1
                && body.get(next) == Some(&marker)
                && body.get(next + 1).is_some_and(|c| !c.is_whitespace())
        })
        .count()
}

/// Where `text` shows a span that Org ends early, never opens or runs into the
/// next, and how: a span running into a quoted word other than a possessive
/// 's, a span holding the end of one span and the start of another, or a
/// marker placed like a closing one that closes no span.
fn findings(text: &str) -> Vec<(usize, String)> {
    let chars: Vec<char> = text.chars().collect();
    let masked = mask(&chars);
    let shown = |from: usize, to: usize| -> String {
        let to = to.min(chars.len());
        chars[from..to]
            .iter()
            .map(|&c| if c == '\n' { ' ' } else { c })
            .collect()
    };
    let word = |at: usize| masked.get(at).copied().is_some_and(is_word);
    let mut covered = vec![false; chars.len()];
    let mut out = Vec::new();
    for (start, end) in spans(&masked) {
        covered[start..end].fill(true);
        let possessive = masked.get(end + 1) == Some(&'s') && !word(end + 2);
        let quoted = match masked.get(end) {
            Some('"') => word(end + 1),
            Some('\'') => word(end + 1) && !possessive,
            _ => false,
        };
        if quoted {
            let what = format!("{} runs into {}", shown(start, end), shown(end, end + 12));
            out.push((start, what));
        }
        let joined = joins(&masked[start + 1..end - 1], masked[start]);
        if joined > 0 {
            let what = format!("{} runs {} spans into one", shown(start, end), joined + 1);
            out.push((start, what));
        }
    }
    for at in 1..masked.len() {
        let closer = (masked[at] == '=' || masked[at] == '~')
            && !covered[at]
            && (is_word(masked[at - 1]) || BEFORE_CLOSER.contains(masked[at - 1]))
            && masked
                .get(at + 1)
                .is_none_or(|&c| c.is_whitespace() || AFTER_CLOSER.contains(c));
        if closer {
            let what = format!("{} closes no span", shown(at.saturating_sub(20), at + 1));
            out.push((at, what));
        }
    }
    out
}

#[test]
fn every_org_span_ends_where_it_is_written() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut docs = Vec::new();
    walk(&root.join("docs/orgmode"), &mut docs);
    let mut checked = 0;
    let mut failures = Vec::new();
    for doc in &docs {
        let text =
            fs::read_to_string(doc).unwrap_or_else(|error| panic!("{}: {error}", doc.display()));
        for (first, unit) in units(&text) {
            let chars: Vec<char> = unit.chars().collect();
            checked += spans(&mask(&chars)).len();
            for (at, what) in findings(&unit) {
                let line = first + chars[..at].iter().filter(|&&c| c == '\n').count();
                failures.push(format!("{}:{line}: {what}", relative(root, doc)));
            }
        }
    }
    assert!(checked > 0, "no spans found under docs/orgmode");
    assert!(
        failures.is_empty(),
        "{} places under docs/orgmode show a span Org ends early, never opens or runs on:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn a_span_org_ends_early_or_never_opens_is_refused() {
    for text in [
        "=history=\"shared\"=, the default",
        "=policy=\"auto\"=, no =grad_fn= and",
        "At =cap >= 50000=, one hop",
        "=minimize(fun, budget=..., replicas=...)= runs",
        "see [[file:catalog.org][descriptors]]/=crate::lattice=)",
        "= pilot_draws_qmc= returns",
    ] {
        assert!(!findings(text).is_empty(), "{text} passes");
    }
    for text in [
        "~history=\"shared\"~, the default",
        "=batch_size=1= or =coverage_shared=True= and alpha=[4,1]",
        "the \"=HmcSaSampler=\", =gsa='s step and T(k+1) <= T(k)",
        "the fragment \\(f(x)= 2\\) is LaTeX",
    ] {
        assert!(findings(text).is_empty(), "{text}: {:?}", findings(text));
    }
}

#[test]
fn spans_org_runs_together_across_a_separator_are_refused() {
    for text in [
        "=worst_bound=/=worst_bound_scaled= decide",
        "=check_accept_path=/\n=require_accept_compatible= and",
        "~ask~/~tell~ on 3.1.1",
        "=a=+=b= or =c=&=d=",
    ] {
        assert!(!findings(text).is_empty(), "{text} passes");
    }
    assert_eq!(
        findings("(=Same=/ =Known=/=New=);"),
        [(1, "=Same=/ =Known=/=New= runs 3 spans into one".to_string())]
    );
    for text in [
        "=worst_bound= / =worst_bound_scaled= decide",
        "=check_accept_path= /\n=require_accept_compatible= and",
        "=worst_bound/worst_bound_scaled= and =s/=/x/=",
    ] {
        assert!(findings(text).is_empty(), "{text}: {:?}", findings(text));
    }
}

#[test]
fn spans_are_read_outside_source_blocks_and_within_one_table_cell() {
    let text =
        "A =x=\nand =y=.\n\n#+begin_src python\nf(x=\"a\")\n#+end_src\n| =a | b= |\n* =Top=\nend";
    let expected = [
        (1, "A =x=\nand =y=."),
        (7, " =a "),
        (7, " b= "),
        (8, "* =Top="),
        (9, "end"),
    ];
    assert_eq!(
        units(text),
        expected.map(|(line, unit)| (line, unit.to_string()))
    );
}
