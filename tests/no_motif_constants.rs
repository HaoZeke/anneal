// Copyright (c) anneal developers
//! Source guard against reintroducing size-tuned or named-motif search branches.
//!
//! The scan is lexical. Comments are dropped, `cfg(test)` items are skipped by
//! brace matching, and the remaining code is cut into statements so that a
//! constant, comparison or match arm split over several lines is still seen.
//! Every statement that names a motif, embeds a benchmark size in an
//! identifier, spells a benchmark cluster size, compares an atom count with a
//! literal, declares a size-named constant, or keys a table by size is a hit.
//! A hit must be claimed by an allowlist entry that pins the path, the needle,
//! the number of statements and a digest of their text.

use std::fs;
use std::path::{Path, PathBuf};

/// Cluster sizes the paper, the benchmark tables or the harness scripts name.
const BENCHMARK_SIZES: &[u64] = &[13, 38, 55, 75, 98, 104, 147, 150];

const MOTIF_WORDS: &[&str] = &[
    "icosahedral",
    "icosahedron",
    "icosahedra",
    "icosahedrally",
    "ico",
    "decahedral",
    "decahedron",
    "decahedra",
    "dodecahedral",
    "dodecahedron",
    "mackay",
    "fivefold",
    "fivefolds",
    "pentagon",
    "pentagons",
    "pentagonal",
    "leary",
    "marks",
    "fcc",
    "hcp",
    "bcc",
    "polytetrahedral",
    "facecentredcubic",
    "facecenteredcubic",
    "motifclass",
    "d5h",
];

const SIZE_WORDS: &[&str] = &[
    "n",
    "np",
    "nat",
    "natom",
    "natoms",
    "npoints",
    "nmol",
    "nmols",
    "nbodies",
    "nmolecules",
    "atoms",
    "points",
    "molecules",
    "bodies",
    "clustersize",
];

const SIZE_NAME_WORDS: &[&str] = &[
    "atoms",
    "atom",
    "points",
    "bodies",
    "body",
    "molecules",
    "molecule",
    "size",
    "sizes",
    "natoms",
    "npoints",
    "volume",
];

const INT_TYPES: &[&str] = &[
    "usize", "isize", "u8", "u16", "u32", "u64", "u128", "i8", "i16", "i32", "i64", "i128",
];

const SPECIES_WORDS: &[&str] = &["covalent", "cordero", "element", "elements", "mendeleev"];

const COMPARISONS: &[&str] = &["<=", ">=", "==", "!=", "<", ">"];

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Word(String),
    Int(u64),
    Float(Option<u64>),
    Text(String),
    Op(String),
}

#[derive(Clone, Copy)]
struct Cell {
    code: char,
    skeleton: char,
    line: usize,
}

struct Unit {
    line: usize,
    code: String,
    in_size_match: bool,
    in_species_match: bool,
}

struct Hit {
    line: usize,
    rule: &'static str,
    text: String,
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Cells of one Rust source with comments removed. `code` keeps string and
/// character literals; `skeleton` blanks them so delimiters can be matched.
fn lex_rust(text: &str) -> Vec<Cell> {
    let chars: Vec<char> = text.chars().collect();
    let mut cells = Vec::new();
    let mut line = 1;
    let mut index = 0;
    let mut block_depth = 0usize;
    let mut in_line_comment = false;
    let mut in_string = false;
    let mut raw_hashes: Option<usize> = None;
    let push = |cells: &mut Vec<Cell>, code: char, skeleton: char, line: usize| {
        cells.push(Cell {
            code,
            skeleton,
            line,
        });
    };
    while index < chars.len() {
        let c = chars[index];
        let next = chars.get(index + 1).copied().unwrap_or('\0');
        if c == '\n' {
            push(&mut cells, '\n', '\n', line);
            line += 1;
            in_line_comment = false;
            index += 1;
            continue;
        }
        if in_line_comment {
            index += 1;
            continue;
        }
        if block_depth > 0 {
            if c == '*' && next == '/' {
                block_depth -= 1;
                index += 2;
            } else if c == '/' && next == '*' {
                block_depth += 1;
                index += 2;
            } else {
                index += 1;
            }
            continue;
        }
        if let Some(hashes) = raw_hashes {
            let closes =
                c == '"' && (0..hashes).all(|offset| chars.get(index + 1 + offset) == Some(&'#'));
            if closes {
                push(&mut cells, '"', '"', line);
                for _ in 0..hashes {
                    push(&mut cells, '#', ' ', line);
                }
                raw_hashes = None;
                index += 1 + hashes;
            } else {
                push(&mut cells, c, ' ', line);
                index += 1;
            }
            continue;
        }
        if in_string {
            if c == '\\' && next != '\n' && next != '\0' {
                push(&mut cells, c, ' ', line);
                push(&mut cells, next, ' ', line);
                index += 2;
            } else if c == '"' {
                in_string = false;
                push(&mut cells, c, '"', line);
                index += 1;
            } else {
                push(&mut cells, c, ' ', line);
                index += 1;
            }
            continue;
        }
        match c {
            '/' if next == '/' => {
                in_line_comment = true;
                index += 2;
            }
            '/' if next == '*' => {
                block_depth = 1;
                index += 2;
            }
            '"' => {
                in_string = true;
                push(&mut cells, c, '"', line);
                index += 1;
            }
            'r' if index == 0 || !is_ident_char(chars[index - 1]) || chars[index - 1] == 'b' => {
                let mut hashes = 0;
                while chars.get(index + 1 + hashes) == Some(&'#') {
                    hashes += 1;
                }
                if chars.get(index + 1 + hashes) == Some(&'"') {
                    push(&mut cells, 'r', ' ', line);
                    for _ in 0..hashes {
                        push(&mut cells, '#', ' ', line);
                    }
                    push(&mut cells, '"', '"', line);
                    raw_hashes = Some(hashes);
                    index += 2 + hashes;
                } else {
                    push(&mut cells, c, c, line);
                    index += 1;
                }
            }
            '\'' => {
                let literal_end = if next == '\\' {
                    chars[index + 2..]
                        .iter()
                        .position(|&candidate| candidate == '\'')
                        .map(|offset| index + 2 + offset)
                } else if chars.get(index + 2) == Some(&'\'') {
                    Some(index + 2)
                } else {
                    None
                };
                if let Some(end) = literal_end {
                    push(&mut cells, '\'', '\'', line);
                    for &inner in &chars[index + 1..end] {
                        push(&mut cells, inner, ' ', line);
                    }
                    push(&mut cells, '\'', '\'', line);
                    index = end + 1;
                } else {
                    push(&mut cells, c, c, line);
                    index += 1;
                }
            }
            _ => {
                push(&mut cells, c, c, line);
                index += 1;
            }
        }
    }
    cells
}

fn skeleton_lines(cells: &[Cell]) -> Vec<String> {
    let mut lines = vec![String::new()];
    for cell in cells {
        if cell.skeleton == '\n' {
            lines.push(String::new());
        } else {
            lines.last_mut().expect("line").push(cell.skeleton);
        }
    }
    lines
}

fn compact(text: &str) -> String {
    text.chars().filter(|c| !c.is_whitespace()).collect()
}

fn is_test_attribute(compacted: &str) -> bool {
    !compacted.contains("not(")
        && (compacted.starts_with("#[cfg(test)]")
            || compacted.starts_with("#[cfg(all(test,")
            || compacted.starts_with("#[test]")
            || compacted.starts_with("#[bench]"))
}

/// Index of the last line of the item whose attribute starts on `start`.
fn item_end(lines: &[String], start: usize) -> usize {
    let mut depth = 0i64;
    let mut item_started = false;
    let mut ends_at_semicolon = false;
    let mut seen_brace = false;
    for (offset, line) in lines[start..].iter().enumerate() {
        let trimmed = line.trim();
        if !item_started && depth == 0 && !trimmed.is_empty() && !trimmed.starts_with("#[") {
            item_started = true;
            let mut words = trimmed.split_whitespace().peekable();
            if let Some(word) = words.peek().copied()
                && word.starts_with("pub")
            {
                if word.starts_with("pub(") && !word.contains(')') {
                    while words.next().is_some_and(|piece| !piece.contains(')')) {}
                } else {
                    words.next();
                }
            }
            let keyword = words.next().unwrap_or("");
            let follower = words.next().unwrap_or("");
            ends_at_semicolon = matches!(keyword, "const" | "static" | "use" | "type" | "let")
                && !matches!(follower, "fn" | "unsafe" | "extern" | "async");
        }
        for c in line.chars() {
            match c {
                '(' | '[' => depth += 1,
                ')' | ']' => depth -= 1,
                '{' => {
                    depth += 1;
                    seen_brace = true;
                }
                '}' => {
                    depth -= 1;
                    if item_started && !ends_at_semicolon && depth == 0 && seen_brace {
                        return start + offset;
                    }
                }
                ';' if item_started && depth == 0 && (ends_at_semicolon || !seen_brace) => {
                    return start + offset;
                }
                _ => {}
            }
        }
    }
    lines.len() - 1
}

fn test_only_lines(cells: &[Cell]) -> Vec<bool> {
    let lines = skeleton_lines(cells);
    let mut skip = vec![false; lines.len()];
    let mut index = 0;
    while index < lines.len() {
        let compacted = compact(&lines[index]);
        if compacted.starts_with("#![cfg(test)]") {
            return vec![true; lines.len()];
        }
        if is_test_attribute(&compacted) {
            let end = item_end(&lines, index);
            for flag in &mut skip[index..=end] {
                *flag = true;
            }
            index = end + 1;
        } else {
            index += 1;
        }
    }
    skip
}

fn mentions_size_word(text: &str) -> bool {
    let mut word = String::new();
    let mut found = false;
    for c in text.chars().chain(std::iter::once(' ')) {
        if is_ident_char(c) {
            word.push(c);
        } else {
            if !word.is_empty() && is_size_ident(&word) {
                found = true;
            }
            word.clear();
        }
    }
    found
}

fn mentions_species_word(text: &str) -> bool {
    identifier_words(text).iter().any(|word| {
        let lower = word.to_ascii_lowercase();
        matches!(
            lower.as_str(),
            "z" | "zs" | "atomic_number" | "atomic_numbers"
        ) || words_of(word)
            .iter()
            .any(|piece| SPECIES_WORDS.contains(&piece.as_str()))
    })
}

fn words_of(identifier: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut previous_lower = false;
    for c in identifier.chars() {
        if c == '_' {
            if !current.is_empty() {
                words.push(std::mem::take(&mut current));
            }
            previous_lower = false;
            continue;
        }
        if c.is_ascii_uppercase() && previous_lower && !current.is_empty() {
            words.push(std::mem::take(&mut current));
        }
        previous_lower = c.is_ascii_lowercase() || c.is_ascii_digit();
        current.push(c.to_ascii_lowercase());
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

fn is_size_ident(identifier: &str) -> bool {
    let words = words_of(identifier);
    SIZE_WORDS.contains(&identifier.to_ascii_lowercase().as_str())
        || words.iter().any(|word| {
            matches!(
                word.as_str(),
                "atoms" | "natoms" | "npoints" | "nmol" | "nmols" | "nbodies" | "nmolecules"
            )
        })
        || (words.len() == 2 && words[0] == "n" && SIZE_WORDS.contains(&words[1].as_str()))
        || (words.len() == 2
            && words[1] == "count"
            && matches!(words[0].as_str(), "atom" | "point" | "molecule" | "body"))
}

fn tokenize(text: &str, python: bool) -> Vec<Tok> {
    let chars: Vec<char> = text.chars().collect();
    let mut tokens = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        let c = chars[index];
        if c.is_whitespace() {
            index += 1;
        } else if c == '"' || (python && c == '\'') {
            let quote = c;
            let mut end = index + 1;
            while end < chars.len() && chars[end] != quote {
                end += if chars[end] == '\\' { 2 } else { 1 };
            }
            let end = end.min(chars.len());
            tokens.push(Tok::Text(chars[index + 1..end].iter().collect()));
            index = (end + 1).min(chars.len().max(end + 1));
        } else if c == '\'' {
            let closing = if chars.get(index + 1) == Some(&'\\') {
                chars[index + 2..]
                    .iter()
                    .position(|&candidate| candidate == '\'')
                    .map(|offset| index + 2 + offset)
            } else if chars.get(index + 2) == Some(&'\'') {
                Some(index + 2)
            } else {
                None
            };
            match closing {
                Some(end) => {
                    tokens.push(Tok::Text(chars[index + 1..end].iter().collect()));
                    index = end + 1;
                }
                None => {
                    tokens.push(Tok::Op("'".into()));
                    index += 1;
                }
            }
        } else if c.is_ascii_digit() {
            let start = index;
            while index < chars.len() && is_ident_char(chars[index]) {
                index += 1;
            }
            let mut run: String = chars[start..index].iter().collect();
            let hex = run.starts_with("0x") || run.starts_with("0X");
            let has_int_suffix = INT_TYPES.iter().any(|suffix| run.ends_with(suffix));
            let mut float = !hex && !has_int_suffix && (run.contains('e') || run.contains('E'));
            if float && matches!(chars.get(index), Some('-' | '+')) {
                index += 1;
                while index < chars.len() && is_ident_char(chars[index]) {
                    index += 1;
                }
            }
            if chars.get(index) == Some(&'.') {
                let after = chars.get(index + 1).copied().unwrap_or(' ');
                if after.is_ascii_digit() {
                    float = true;
                    index += 1;
                    while index < chars.len()
                        && (is_ident_char(chars[index])
                            || (matches!(chars[index], '-' | '+')
                                && matches!(chars[index - 1], 'e' | 'E')))
                    {
                        index += 1;
                    }
                } else if after != '.' && !is_ident_start(after) {
                    float = true;
                    index += 1;
                }
            }
            run = chars[start..index].iter().collect();
            if float || run.ends_with("f32") || run.ends_with("f64") {
                let benchmark = run
                    .strip_suffix("f32")
                    .or_else(|| run.strip_suffix("f64"))
                    .unwrap_or(run.as_str())
                    .replace('_', "")
                    .parse::<f64>()
                    .ok()
                    .and_then(|value| {
                        let rounded = value.round();
                        ((value - rounded).abs() <= f64::EPSILON * 32.0).then_some(rounded as u64)
                    })
                    .filter(|value| BENCHMARK_SIZES.contains(value));
                tokens.push(Tok::Float(benchmark));
            } else {
                let mut digits = run.replace('_', "");
                for suffix in INT_TYPES {
                    if let Some(stripped) = digits.strip_suffix(suffix) {
                        digits = stripped.to_string();
                        break;
                    }
                }
                let value = if let Some(rest) = digits
                    .strip_prefix("0x")
                    .or_else(|| digits.strip_prefix("0X"))
                {
                    u64::from_str_radix(rest, 16).ok()
                } else if let Some(rest) = digits.strip_prefix("0b") {
                    u64::from_str_radix(rest, 2).ok()
                } else if let Some(rest) = digits.strip_prefix("0o") {
                    u64::from_str_radix(rest, 8).ok()
                } else {
                    digits.parse::<u64>().ok()
                };
                tokens.push(value.map_or(Tok::Float(None), Tok::Int));
            }
        } else if is_ident_start(c) {
            let start = index;
            while index < chars.len() && is_ident_char(chars[index]) {
                index += 1;
            }
            tokens.push(Tok::Word(chars[start..index].iter().collect()));
        } else {
            let rest: String = chars[index..chars.len().min(index + 3)].iter().collect();
            let operator = ["<<=", ">>=", "..="]
                .iter()
                .chain(
                    [
                        "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "=>", "->", "..", "::",
                    ]
                    .iter(),
                )
                .find(|operator| rest.starts_with(**operator))
                .map_or_else(|| c.to_string(), |operator| (*operator).to_string());
            index += operator.chars().count();
            tokens.push(Tok::Op(operator));
        }
    }
    tokens
}

fn int_at(tokens: &[Tok], index: usize) -> Option<u64> {
    match tokens.get(index) {
        Some(Tok::Int(value)) => Some(*value),
        Some(Tok::Op(star)) if star == "*" || star == "&" => match tokens.get(index + 1) {
            Some(Tok::Int(value)) => Some(*value),
            _ => None,
        },
        _ => None,
    }
}

fn is_delimiter(token: &Tok) -> bool {
    matches!(token, Tok::Op(op) if matches!(op.as_str(),
        "&&" | "||" | "(" | "," | "{" | "}" | ";" | "=" | "=>" | "[" | "!"))
        || matches!(token, Tok::Word(word) if matches!(word.as_str(),
            "if" | "while" | "return" | "and" | "or" | "not" | "assert" | "else" | "elif"))
}

fn window_mentions_size(tokens: &[Tok], range: impl Iterator<Item = usize>) -> bool {
    for index in range {
        match &tokens[index] {
            Tok::Word(word) if is_size_ident(word) => return true,
            token if is_delimiter(token) => return false,
            _ => {}
        }
    }
    false
}

fn compares_size_with_literal(tokens: &[Tok], minimum: u64) -> bool {
    for (index, token) in tokens.iter().enumerate() {
        let Tok::Op(op) = token else { continue };
        if !COMPARISONS.contains(&op.as_str()) {
            continue;
        }
        let right_literal = int_at(tokens, index + 1).is_some_and(|value| value >= minimum);
        let left_literal = index > 0
            && matches!(&tokens[index - 1], Tok::Int(value) if *value >= minimum)
            && (index == 1 || is_delimiter(&tokens[index - 2]));
        if right_literal && window_mentions_size(tokens, (0..index).rev()) {
            return true;
        }
        if left_literal && window_mentions_size(tokens, index + 1..tokens.len()) {
            return true;
        }
    }
    false
}

fn size_assignment(tokens: &[Tok], minimum: u64) -> bool {
    for (index, token) in tokens.iter().enumerate() {
        let Tok::Word(name) = token else { continue };
        if !is_size_ident(name) {
            continue;
        }
        let mut cursor = index + 1;
        if matches!(tokens.get(cursor), Some(Tok::Op(colon)) if colon == ":")
            && matches!(tokens.get(cursor + 1), Some(Tok::Int(value)) if *value >= minimum)
        {
            return true;
        }
        if matches!(tokens.get(cursor), Some(Tok::Op(colon)) if colon == ":") {
            cursor += 1;
            while matches!(tokens.get(cursor), Some(Tok::Word(_)) | Some(Tok::Op(_)))
                && !matches!(tokens.get(cursor), Some(Tok::Op(op)) if op == "=")
                && cursor < index + 6
            {
                cursor += 1;
            }
        }
        if matches!(tokens.get(cursor), Some(Tok::Op(equals)) if equals == "=")
            && matches!(tokens.get(cursor + 1), Some(Tok::Int(value)) if *value >= minimum)
        {
            return true;
        }
    }
    false
}

fn declared_constant(tokens: &[Tok]) -> Option<(String, usize, usize)> {
    let mut index = 0;
    while matches!(tokens.get(index), Some(Tok::Op(op)) if op == "#") {
        let mut depth = 0i64;
        index += 1;
        while let Some(token) = tokens.get(index) {
            if matches!(token, Tok::Op(op) if op == "[") {
                depth += 1;
            } else if matches!(token, Tok::Op(op) if op == "]") {
                depth -= 1;
                if depth == 0 {
                    index += 1;
                    break;
                }
            }
            index += 1;
        }
    }
    if matches!(tokens.get(index), Some(Tok::Word(word)) if word == "pub") {
        index += 1;
        if matches!(tokens.get(index), Some(Tok::Op(op)) if op == "(") {
            while !matches!(tokens.get(index), Some(Tok::Op(op)) if op == ")")
                && index < tokens.len()
            {
                index += 1;
            }
            index += 1;
        }
    }
    match tokens.get(index) {
        Some(Tok::Word(word)) if word == "const" || word == "static" => {}
        _ => return None,
    }
    index += 1;
    if matches!(tokens.get(index), Some(Tok::Word(word)) if word == "mut") {
        index += 1;
    }
    let Some(Tok::Word(name)) = tokens.get(index) else {
        return None;
    };
    if name == "fn" || name == "unsafe" || name == "extern" || name == "async" {
        return None;
    }
    let equals = tokens
        .iter()
        .skip(index)
        .position(|token| matches!(token, Tok::Op(op) if op == "="))?
        + index;
    Some((name.clone(), index, equals))
}

fn constant_findings(tokens: &[Tok], minimum: u64) -> Option<&'static str> {
    let (name, name_index, equals) = declared_constant(tokens)?;
    let declared_type = &tokens[name_index + 1..equals];
    let value = &tokens[equals + 1..];
    let has_int_type = declared_type
        .iter()
        .any(|token| matches!(token, Tok::Word(word) if INT_TYPES.contains(&word.as_str())));
    if !has_int_type {
        return None;
    }
    let has_big_int = value
        .iter()
        .any(|token| matches!(token, Tok::Int(value) if *value >= minimum));
    let name_words = words_of(&name);
    if has_big_int
        && name_words
            .iter()
            .any(|word| SIZE_NAME_WORDS.contains(&word.as_str()))
    {
        return Some("size-named constant");
    }
    let tuple_keyed = declared_type.windows(2).any(|pair| {
        matches!(&pair[0], Tok::Op(op) if op == "(")
            && matches!(&pair[1], Tok::Word(word) if INT_TYPES.contains(&word.as_str()))
    });
    let first_key = value.windows(3).any(|triple| {
        matches!(&triple[0], Tok::Op(op) if op == "[" || op == "(")
            && matches!(&triple[1], Tok::Op(op) if op == "(")
            && matches!(&triple[2], Tok::Int(value) if *value >= minimum)
    });
    if tuple_keyed && first_key {
        return Some("size-keyed table");
    }
    None
}

fn embedded_benchmark_size(word: &str) -> bool {
    word.split('_').any(embedded_benchmark_piece)
}

fn embedded_benchmark_piece(word: &str) -> bool {
    let lower = word.to_ascii_lowercase();
    if !lower.starts_with(|c: char| c.is_ascii_alphabetic()) {
        return false;
    }
    let digits_start = lower
        .find(|c: char| c.is_ascii_digit())
        .unwrap_or(lower.len());
    let digits = &lower[digits_start..];
    (digits_start >= 2 || lower.starts_with('n') && digits_start == 1)
        && digits_start < lower.len()
        && digits.chars().all(|c| c.is_ascii_digit())
        && digits
            .parse::<u64>()
            .is_ok_and(|value| BENCHMARK_SIZES.contains(&value))
}

fn identifier_words(text: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut run = String::new();
    for c in text.chars().chain(std::iter::once(' ')) {
        if is_ident_char(c) {
            run.push(c);
        } else if !run.is_empty() {
            found.push(std::mem::take(&mut run));
        }
    }
    found
}

fn motif_in(text: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    for run in identifier_words(text) {
        let lowered = run.to_ascii_lowercase();
        if MOTIF_WORDS.contains(&lowered.as_str()) {
            return Some(lowered);
        }
        let mut pieces = words_of(&run);
        pieces.extend(lowered.split('_').map(str::to_string));
        if pieces
            .iter()
            .any(|piece| MOTIF_WORDS.contains(&piece.as_str()))
        {
            return Some(lowered);
        }
        let joined: Vec<&str> = lowered.split('_').collect();
        if joined
            .windows(2)
            .any(|pair| pair[0] == "five" && pair[1] == "fold")
            || lowered.contains("fivefold")
            || lowered.contains("icosahedr")
            || lowered.contains("decahedr")
            || lowered.contains("pentagon")
            || lowered.contains("motifclass")
            || lowered.contains("facecentredcubic")
            || lowered.contains("5fold")
        {
            return Some(lowered);
        }
    }
    if lower.contains("d_5h") || lower.contains("cyclic_about_z(5)") {
        return Some("fixed fivefold group".into());
    }
    None
}

fn pattern_positions(tokens: &[Tok]) -> Vec<bool> {
    let mut positions = vec![false; tokens.len()];
    if !tokens
        .iter()
        .any(|token| matches!(token, Tok::Op(op) if op == "=>"))
    {
        return positions;
    }
    let mut at_pattern = true;
    let mut depth = 0i64;
    for (index, token) in tokens.iter().enumerate() {
        match token {
            Tok::Op(op) if op == "(" || op == "[" => depth += 1,
            Tok::Op(op) if op == ")" || op == "]" => depth -= 1,
            Tok::Op(op) if op == "=>" => at_pattern = false,
            Tok::Op(op) if op == "," && depth == 0 => at_pattern = true,
            _ => positions[index] = at_pattern,
        }
    }
    positions
}

fn benchmark_literal_context(tokens: &[Tok], index: usize, in_pattern: bool) -> bool {
    let op_is = |token: Option<&Tok>, set: &[&str]| matches!(token, Some(Tok::Op(op)) if set.contains(&op.as_str()));
    let previous = index.checked_sub(1).map(|at| &tokens[at]);
    let next = tokens.get(index + 1);
    if op_is(previous, &["<<", ">>"]) {
        return false;
    }
    in_pattern
        || op_is(previous, COMPARISONS)
        || op_is(next, COMPARISONS)
        || op_is(previous, &["..", "..="])
        || (op_is(previous, &["="]) && next.is_none())
        || op_is(next, &["..", "..=", "=>"])
        || matches!(previous, Some(Tok::Word(word)) if word == "return")
        || benchmark_min_max_call(tokens, index)
        || benchmark_vec_repeat(tokens, index)
}

fn benchmark_min_max_call(tokens: &[Tok], index: usize) -> bool {
    let Some(open) = index.checked_sub(1).and_then(|at| tokens.get(at)) else {
        return false;
    };
    let Some(name) = index.checked_sub(2).and_then(|at| tokens.get(at)) else {
        return false;
    };
    let Some(dot) = index.checked_sub(3).and_then(|at| tokens.get(at)) else {
        return false;
    };
    matches!(open, Tok::Op(open) if open == "(")
        && matches!(name, Tok::Word(word) if matches!(word.as_str(), "min" | "max"))
        && matches!(dot, Tok::Op(dot) if dot == ".")
}

fn benchmark_vec_repeat(tokens: &[Tok], index: usize) -> bool {
    let Some(previous) = index.checked_sub(1).and_then(|at| tokens.get(at)) else {
        return false;
    };
    if !matches!(previous, Tok::Op(op) if op == ";") {
        return false;
    }
    let mut depth = 0i64;
    let mut cursor = index as i64 - 1;
    while cursor >= 0 {
        match &tokens[cursor as usize] {
            Tok::Op(op) if op == "]" => depth += 1,
            Tok::Op(op) if op == "[" => {
                if depth == 0 {
                    let Some(bang_index) = (cursor as usize).checked_sub(1) else {
                        return false;
                    };
                    let Some(name_index) = (cursor as usize).checked_sub(2) else {
                        return false;
                    };
                    return matches!(tokens.get(bang_index), Some(Tok::Op(bang)) if bang == "!")
                        && matches!(tokens.get(name_index), Some(Tok::Word(word)) if word == "vec");
                }
                depth -= 1;
            }
            _ => {}
        }
        cursor -= 1;
    }
    false
}

fn benchmark_system_name(text: &str) -> bool {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|piece| !piece.is_empty())
        .any(|piece| {
            let lower = piece.to_ascii_lowercase();
            ["lj", "au", "pt"].iter().any(|prefix| {
                lower.strip_prefix(prefix).is_some_and(|digits| {
                    !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit())
                }) && lower[prefix.len()..]
                    .parse::<u64>()
                    .is_ok_and(|value| BENCHMARK_SIZES.contains(&value))
            })
        })
}

fn unit_findings(unit: &Unit, python: bool) -> Option<&'static str> {
    if motif_in(&unit.code).is_some() {
        return Some("named motif");
    }
    let tokens = tokenize(&unit.code, python);
    let in_pattern = pattern_positions(&tokens);
    let constant = declared_constant(&tokens).is_some_and(|(_, name_index, equals)| {
        tokens[name_index + 1..equals]
            .iter()
            .any(|token| matches!(token, Tok::Word(word) if INT_TYPES.contains(&word.as_str())))
    });
    let benchmark_sizes = tokens
        .iter()
        .filter(|token| matches!(token, Tok::Int(value) if BENCHMARK_SIZES.contains(value)))
        .count();
    for (index, token) in tokens.iter().enumerate() {
        match token {
            Tok::Word(word) if embedded_benchmark_size(word) => {
                return Some("benchmark size in identifier");
            }
            Tok::Word(word) => {
                let words = words_of(word);
                if words.len() > 1
                    && words
                        .iter()
                        .any(|piece| piece == "lj" || piece == "au" || piece == "pt")
                    && words.iter().any(|piece| {
                        piece
                            .parse::<u64>()
                            .is_ok_and(|value| BENCHMARK_SIZES.contains(&value))
                    })
                {
                    return Some("benchmark size in identifier");
                }
            }
            Tok::Int(value)
                if BENCHMARK_SIZES.contains(value)
                    && (constant
                        || benchmark_sizes >= 2
                        || benchmark_literal_context(&tokens, index, in_pattern[index])) =>
            {
                return Some("benchmark cluster size literal");
            }
            Tok::Float(Some(value))
                if BENCHMARK_SIZES.contains(value)
                    && (constant
                        || benchmark_sizes >= 2
                        || benchmark_literal_context(&tokens, index, in_pattern[index])) =>
            {
                return Some("benchmark cluster size literal");
            }
            Tok::Text(text) if benchmark_system_name(text) => {
                return Some("benchmark size in system name");
            }
            _ => {}
        }
    }
    if let Some(rule) = constant_findings(&tokens, 5) {
        return Some(rule);
    }
    if compares_size_with_literal(&tokens, 5) {
        return Some("size compared with a literal");
    }
    if size_assignment(&tokens, 5) {
        return Some("size variable fixed to a literal");
    }
    if unit.in_size_match && arm_pattern_has_int(&tokens, 5) {
        return Some("match arm keyed by size");
    }
    if unit.in_species_match && arm_pattern_has_int(&tokens, 1) {
        return Some("match arm keyed by atomic number");
    }
    if declared_constant(&tokens).is_some()
        && tokens
            .iter()
            .filter(|token| matches!(token, Tok::Float(_) | Tok::Int(_)))
            .count()
            >= 40
    {
        return Some("bulk numeric table");
    }
    group_size_hit(&unit.code).then_some("group cardinality compared with a literal")
}

fn arm_pattern_has_int(tokens: &[Tok], minimum: u64) -> bool {
    let mut at_pattern = true;
    let mut depth = 0i64;
    for token in tokens {
        match token {
            Tok::Op(op) if op == "(" || op == "[" => depth += 1,
            Tok::Op(op) if op == ")" || op == "]" => depth -= 1,
            Tok::Op(op) if op == "=>" => at_pattern = false,
            Tok::Op(op) if op == "," && depth == 0 => at_pattern = true,
            Tok::Int(value) if at_pattern && *value >= minimum => return true,
            _ => {}
        }
    }
    false
}

fn group_size_hit(code: &str) -> bool {
    let compacted = compact(code);
    let Some(rest) = compacted.split("groups[0].len()").nth(1) else {
        return false;
    };
    COMPARISONS.iter().any(|comparison| {
        let Some(rest) = rest.strip_prefix(comparison) else {
            return false;
        };
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        digits.parse::<usize>().is_ok_and(|number| number >= 4)
    })
}

fn after_match_keyword(skeleton: &str) -> Option<&str> {
    skeleton.match_indices("match").find_map(|(start, word)| {
        let end = start + word.len();
        let before = skeleton[..start].chars().next_back();
        let after = skeleton[end..].chars().next();
        (!before.is_some_and(is_ident_char) && !after.is_some_and(is_ident_char))
            .then(|| &skeleton[end..])
    })
}

fn units_of(cells: &[Cell], skip: &[bool]) -> Vec<Unit> {
    let mut units = Vec::new();
    let mut stack: Vec<(bool, bool)> = Vec::new();
    let mut code = String::new();
    let mut skeleton = String::new();
    let mut first_line = 0usize;
    let mut bracket = 0i64;
    let flush = |units: &mut Vec<Unit>,
                 code: &mut String,
                 skeleton: &mut String,
                 first_line: &mut usize,
                 context: (bool, bool)| {
        let text = code.split_whitespace().collect::<Vec<_>>().join(" ");
        if !text.is_empty() {
            units.push(Unit {
                line: *first_line,
                code: text,
                in_size_match: context.0,
                in_species_match: context.1,
            });
        }
        code.clear();
        skeleton.clear();
        *first_line = 0;
    };
    for cell in cells {
        if skip.get(cell.line - 1).copied().unwrap_or(false) {
            continue;
        }
        let delimiter = bracket == 0 && matches!(cell.skeleton, '{' | '}' | ';');
        if cell.skeleton == '[' {
            bracket += 1;
        } else if cell.skeleton == ']' {
            bracket -= 1;
        }
        if !delimiter {
            if first_line == 0 && !cell.code.is_whitespace() {
                first_line = cell.line;
            }
            code.push(cell.code);
            skeleton.push(cell.skeleton);
            continue;
        }
        let top = stack.last().copied().unwrap_or((false, false));
        match cell.skeleton {
            '{' => {
                let scrutinee = after_match_keyword(&skeleton);
                let opens = (
                    scrutinee.is_some_and(mentions_size_word),
                    scrutinee.is_some_and(mentions_species_word),
                );
                flush(&mut units, &mut code, &mut skeleton, &mut first_line, top);
                stack.push(opens);
            }
            '}' => {
                flush(&mut units, &mut code, &mut skeleton, &mut first_line, top);
                stack.pop();
            }
            _ => flush(&mut units, &mut code, &mut skeleton, &mut first_line, top),
        }
    }
    flush(
        &mut units,
        &mut code,
        &mut skeleton,
        &mut first_line,
        (false, false),
    );
    units
}

fn rust_units(text: &str) -> Vec<Unit> {
    let cells = lex_rust(text);
    let skip = test_only_lines(&cells);
    units_of(&cells, &skip)
}

fn constant_name(unit: &Unit) -> Option<String> {
    declared_constant(&tokenize(&unit.code, false)).map(|(name, _, _)| name)
}

fn mentions_word(unit: &Unit, name: &str) -> bool {
    tokenize(&unit.code, false)
        .iter()
        .any(|token| matches!(token, Tok::Word(word) if word == name))
}

/// Hits for several sources at once. A constant that is itself a hit as a
/// size taints every other statement that reads it.
fn rust_hits(sources: &[(String, String)]) -> Vec<(String, Hit)> {
    let units: Vec<(&str, Unit)> = sources
        .iter()
        .flat_map(|(path, text)| {
            rust_units(text)
                .into_iter()
                .map(move |unit| (path.as_str(), unit))
        })
        .collect();
    let first: Vec<Option<&'static str>> = units
        .iter()
        .map(|(_, unit)| unit_findings(unit, false))
        .collect();
    let mut tainted: Vec<String> = Vec::new();
    for ((_, unit), rule) in units.iter().zip(&first) {
        if matches!(
            rule,
            Some("size-named constant" | "benchmark cluster size literal")
        ) && let Some(name) = constant_name(unit)
            && name.len() > 3
            && !tainted.contains(&name)
        {
            tainted.push(name);
        }
    }
    let mut hits = Vec::new();
    for ((path, unit), rule) in units.iter().zip(first) {
        let rule = rule.or_else(|| {
            let own = constant_name(unit);
            tainted
                .iter()
                .any(|name| own.as_ref() != Some(name) && mentions_word(unit, name))
                .then_some("reads a size constant")
        });
        if let Some(rule) = rule {
            hits.push((
                (*path).to_string(),
                Hit {
                    line: unit.line,
                    rule,
                    text: unit.code.clone(),
                },
            ));
        }
    }
    hits
}

fn python_units(text: &str) -> Vec<Unit> {
    let mut units = Vec::new();
    let mut in_triple: Option<&str> = None;
    let mut pending = String::new();
    let mut pending_line = 0usize;
    let mut depth = 0i64;
    for (index, raw) in text.lines().enumerate() {
        let mut code = String::new();
        let mut rest = raw;
        loop {
            if let Some(quote) = in_triple {
                match rest.find(quote) {
                    Some(end) => {
                        rest = &rest[end + 3..];
                        in_triple = None;
                    }
                    None => break,
                }
            }
            let triple = ["\"\"\"", "'''"]
                .iter()
                .filter_map(|quote| rest.find(quote).map(|at| (at, *quote)))
                .min_by_key(|(at, _)| *at);
            let hash = rest.find('#');
            match (triple, hash) {
                (Some((at, quote)), hash) if hash.is_none_or(|h| at < h) => {
                    code.push_str(&rest[..at]);
                    rest = &rest[at + 3..];
                    in_triple = Some(quote);
                }
                (_, Some(h)) => {
                    let quoted = rest[..h].matches('"').count() % 2 == 1
                        || rest[..h].matches('\'').count() % 2 == 1;
                    if quoted {
                        code.push_str(rest);
                    } else {
                        code.push_str(&rest[..h]);
                    }
                    break;
                }
                _ => {
                    code.push_str(rest);
                    break;
                }
            }
        }
        if code.trim().is_empty() {
            continue;
        }
        if pending.is_empty() {
            pending_line = index + 1;
        }
        for c in code.chars() {
            match c {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                _ => {}
            }
        }
        pending.push(' ');
        pending.push_str(code.trim());
        if depth <= 0 && !code.trim_end().ends_with('\\') {
            let text = pending.split_whitespace().collect::<Vec<_>>().join(" ");
            units.push(Unit {
                line: pending_line,
                code: text,
                in_size_match: false,
                in_species_match: false,
            });
            pending.clear();
            depth = 0;
        }
    }
    units
}

fn has_benchmark_size(tokens: &[Tok]) -> bool {
    tokens
        .iter()
        .any(|token| matches!(token, Tok::Int(value) if BENCHMARK_SIZES.contains(value)))
}

fn python_hits(text: &str, library: bool) -> Vec<Hit> {
    python_units(text)
        .into_iter()
        .filter_map(|unit| {
            let rule = if library {
                unit_findings(&unit, true)
            } else {
                let tokens = tokenize(&unit.code, true);
                (compares_size_with_literal(&tokens, BENCHMARK_SIZES[0])
                    && has_benchmark_size(&tokens)
                    || size_assignment(&tokens, BENCHMARK_SIZES[0]) && has_benchmark_size(&tokens))
                .then_some("benchmark size default or comparison in a harness")
            };
            rule.map(|rule| Hit {
                line: unit.line,
                rule,
                text: unit.code,
            })
        })
        .collect()
}

struct Allow {
    path: &'static str,
    needle: &'static str,
    count: usize,
    digest: u64,
    reason: &'static str,
}

fn fnv(lines: &[&str]) -> u64 {
    // FNV-1a is sufficient here: this digest only guards accidental
    // statement drift in pinned allowlist entries, not adversarial collision
    // resistance. Path, needle, and statement count already scope each pin.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for line in lines {
        for byte in line.bytes().chain(std::iter::once(b'\n')) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}

const ALLOW: &[Allow] = &[
    Allow {
        path: "src/methods/cluster_hopping/config.rs",
        needle: "if n_points < 98",
        count: 2,
        digest: 0xd5b0b1f9b5ad6699,
        reason: "known surface-size branch; bc-a84a44f0 is replacing it",
    },
    Allow {
        path: "src/methods/cluster_hopping/config.rs",
        needle: "groups[0].len() >= 4",
        count: 1,
        digest: 0x63d24cd18bb97b63,
        reason: "existing lone-flexible-group SOAP switch; replacement is audited",
    },
    // False positive (a): minimum 3D simplex (4 vertices) plus 2 movable points for non-degenerate hole competition.
    Allow {
        path: "src/methods/cluster_hopping/general.rs",
        needle: "if n < 6",
        count: 1,
        digest: 0x904c3f365d2dcd35,
        reason: "false positive (a): 3D simplex plus two movable sites minimum array precondition",
    },
    // False positive (a): experimental single-bond covalent radii table (Cordero et al. 2008) of array length 97.
    Allow {
        path: "src/methods/cluster_hopping/moves.rs",
        needle: "const r: [f64; 97]",
        count: 1,
        digest: 0x745bdcd5dc2d8836,
        reason: "false positive (a): physical Cordero covalent radii table for Z = 1..=96 (array length 97)",
    },
    // False positive (a): diagnostic-only stderr trace counter; never read by search steering.
    Allow {
        path: "src/methods/cluster_hopping.rs",
        needle: "fcc",
        count: 6,
        digest: 0x6470f39db06ae259,
        reason: "false positive (a): diagnostic stderr trace counter; comments prohibit downstream steering",
    },
    // False positive (a): diagnostic-only stderr trace counter; never read by search steering.
    Allow {
        path: "src/methods/cluster_hopping.rs",
        needle: "hcp",
        count: 6,
        digest: 0xb074ca6aa4571b07,
        reason: "false positive (a): diagnostic stderr trace counter; comments prohibit downstream steering",
    },
    // False positive (a): diagnostic-only stderr trace counter; never read by search steering.
    Allow {
        path: "src/methods/cluster_hopping.rs",
        needle: "mut ico",
        count: 2,
        digest: 0x969e1e6c622b65a7,
        reason: "false positive (a): diagnostic stderr trace counter; comments prohibit downstream steering",
    },
    // False positive (a): diagnostic-only PTM label tally on stderr trace; never selects a move.
    Allow {
        path: "src/methods/cluster_hopping.rs",
        needle: "Template::FaceCentredCubic",
        count: 2,
        digest: 0x8cd21a29f7a26c1d,
        reason: "false positive (a): diagnostic PTM trace classification; never steers search",
    },
    // False positive (a): diagnostic-only PTM label tally on stderr trace; never selects a move.
    Allow {
        path: "src/methods/cluster_hopping.rs",
        needle: "Template::Icosahedral",
        count: 2,
        digest: 0x8cd21a29f7a26c1d,
        reason: "false positive (a): diagnostic PTM trace classification; never steers search",
    },
    // False positive (a): Steinhardt et al. (1983) mathematical spherical-harmonic invariant for diagnostic tests.
    Allow {
        path: "src/morphology.rs",
        needle: "ICOSAHEDRAL",
        count: 1,
        digest: 0x0659cab714e6814b,
        reason: "false positive (a): Steinhardt spherical-harmonic invariant constant for diagnostic reference",
    },
    // False positive (a): Steinhardt et al. (1983) mathematical spherical-harmonic invariant for diagnostic tests.
    Allow {
        path: "src/morphology.rs",
        needle: "FCC",
        count: 1,
        digest: 0xd63e597ab73804e8,
        reason: "false positive (a): Steinhardt spherical-harmonic invariant constant for diagnostic reference",
    },
    // False positive (a): Steinhardt et al. (1983) mathematical spherical-harmonic invariant for diagnostic tests.
    Allow {
        path: "src/morphology.rs",
        needle: "HCP",
        count: 1,
        digest: 0x031825ff84a001a0,
        reason: "false positive (a): Steinhardt spherical-harmonic invariant constant for diagnostic reference",
    },
    // False positive (a): published Morse reference energies read only by post-run benchmark hit reporting.
    Allow {
        path: "src/potentials.rs",
        needle: "(6, 55) =>",
        count: 1,
        digest: 0xe25ed054a1c6a8e6,
        reason: "false positive (a): published Morse ground-state reference energies for post-run hit reporting",
    },
    // False positive (a): crystallographic polyhedron geometry for diagnostic PTM classification only.
    Allow {
        path: "src/structure.rs",
        needle: "facecentredcubic",
        count: 5,
        digest: 0x1553b80cd5da31ab,
        reason: "false positive (a): crystallographic PTM polyhedron geometry used only for diagnostic classification",
    },
    // False positive (a): crystallographic polyhedron geometry for diagnostic PTM classification only.
    Allow {
        path: "src/structure.rs",
        needle: "icosahedral",
        count: 5,
        digest: 0xf7f8caf4f67a7903,
        reason: "false positive (a): crystallographic PTM polyhedron geometry used only for diagnostic classification",
    },
    // False positive (a): sliding a 4-atom planar sheet across a distinct 4-atom backing layer requires 8 coordinates.
    Allow {
        path: "src/twin.rs",
        needle: "if n < 8",
        count: 2,
        digest: 0x9b65d1a69024f003,
        reason: "false positive (a): two four-atom layers minimum array-length precondition",
    },
];

const PYTHON_ALLOW: &[Allow] = &[
    Allow {
        path: "experiments/scripts/superbasin_paired.py",
        needle: "n = 75 if",
        count: 1,
        digest: 0x3bd76586d73f89d5,
        reason: "experiment infers the benchmark size from a directory name",
    },
    Allow {
        path: "scripts/ase_lj75/run_one.py",
        needle: "N = 75",
        count: 1,
        digest: 0x5f41751b4fdf8260,
        reason: "competitor harness fixed to the named LJ75 benchmark system",
    },
    Allow {
        path: "scripts/lj38_paired/agox_documented.py",
        needle: "n = 38",
        count: 1,
        digest: 0x5dba185f91b50765,
        reason: "competitor harness fixed to the named LJ38 benchmark system",
    },
    Allow {
        path: "scripts/pimc_lj38/run_campaign.py",
        needle: "n == 38",
        count: 3,
        digest: 0x672908dda71d23db,
        reason: "competitor harness branches on the named LJ38 benchmark system",
    },
    Allow {
        path: "scripts/pimc_lj38/run_campaign.py",
        needle: "n: int = 38",
        count: 5,
        digest: 0x0ff0d7090a343440,
        reason: "competitor harness defaults to the named LJ38 benchmark system",
    },
];

fn collect_files(root: &Path, extension: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries {
        let path = entry.expect("source entry").path();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if path.is_dir() {
            if !matches!(name, "__pycache__" | "target" | ".git") {
                collect_files(&path, extension, out);
            }
        } else if path.extension().is_some_and(|found| found == extension)
            && !name.starts_with("test_")
            && !name.ends_with("_test.py")
            && name != "conftest.py"
        {
            out.push(path);
        }
    }
}

fn files_under(roots: &[&Path], extension: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for root in roots {
        collect_files(root, extension, &mut out);
    }
    out.sort();
    out
}

/// Per-entry check: every hit is claimed, and each entry matches exactly the
/// statements it pinned.
fn audit(hits: &[(String, Hit)], allow: &[Allow]) -> (Vec<String>, Vec<String>) {
    let mut claimed: Vec<Vec<&str>> = vec![Vec::new(); allow.len()];
    let mut unreviewed = Vec::new();
    for (path, hit) in hits {
        let lower = hit.text.to_ascii_lowercase();
        let mut allowed = false;
        for (slot, entry) in allow.iter().enumerate() {
            assert!(
                !entry.reason.trim().is_empty(),
                "allowlist reasons are mandatory"
            );
            if path == entry.path && lower.contains(&entry.needle.to_ascii_lowercase()) {
                claimed[slot].push(hit.text.as_str());
                allowed = true;
            }
        }
        if !allowed {
            unreviewed.push(format!("{path}:{}: [{}] {}", hit.line, hit.rule, hit.text));
        }
    }
    let stale = allow
        .iter()
        .zip(claimed)
        .filter(|(entry, lines)| lines.len() != entry.count || fnv(lines) != entry.digest)
        .map(|(entry, lines)| {
            format!(
                "{}: `{}` matched {} statements with digest {:#018x}, pinned {} with {:#018x} ({})",
                entry.path,
                entry.needle,
                lines.len(),
                fnv(&lines),
                entry.count,
                entry.digest,
                entry.reason
            )
        })
        .collect();
    (unreviewed, stale)
}

fn relative_path(root: &Path, file: &Path) -> String {
    file.strip_prefix(root)
        .expect("source below root")
        .to_string_lossy()
        .replace('\\', "/")
}

fn scan_rust_tree(root: &Path) -> Vec<(String, Hit)> {
    let sources: Vec<(String, String)> =
        files_under(&[&root.join("src"), &root.join("crates")], "rs")
            .iter()
            .map(|file| (relative_path(root, file), file))
            // Generated Cap'n Proto bindings are machine output and can
            // contain benchmark identifiers and dense tables unrelated to
            // search policy logic.
            .filter(|(relative, _)| !relative.ends_with("_capnp.rs"))
            .map(|(relative, file)| (relative, fs::read_to_string(file).expect("read source")))
            .collect();
    rust_hits(&sources)
}

fn scan_python_tree(root: &Path) -> Vec<(String, Hit)> {
    let mut hits = Vec::new();
    let library = root.join("python");
    for (roots, is_library) in [
        (vec![library.as_path()], true),
        (
            vec![
                &root.join("scripts") as &Path,
                &root.join("experiments"),
                &root.join("bench"),
            ],
            false,
        ),
    ] {
        for file in files_under(&roots, "py") {
            let relative = relative_path(root, &file);
            // Test fixtures intentionally mirror benchmark labels/literals.
            if relative.contains("/test/") || relative.contains("/tests/") {
                continue;
            }
            let text = fs::read_to_string(&file).expect("read python source");
            hits.extend(
                python_hits(&text, is_library)
                    .into_iter()
                    .map(|hit| (relative.clone(), hit)),
            );
        }
    }
    hits
}

#[test]
fn search_source_has_no_unreviewed_size_or_motif_branches() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let (unreviewed, stale) = audit(&scan_rust_tree(root), ALLOW);
    assert!(
        stale.is_empty(),
        "stale no-motif allowlist entries:\n{}",
        stale.join("\n")
    );
    assert!(
        unreviewed.is_empty(),
        "unreviewed size or named-motif search branches:\n{}",
        unreviewed.join("\n")
    );
}

#[test]
fn python_sources_have_no_unreviewed_size_or_motif_defaults() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let (unreviewed, stale) = audit(&scan_python_tree(root), PYTHON_ALLOW);
    assert!(
        stale.is_empty(),
        "stale python allowlist entries:\n{}",
        stale.join("\n")
    );
    assert!(
        unreviewed.is_empty(),
        "unreviewed size or named-motif python defaults:\n{}",
        unreviewed.join("\n")
    );
}

fn hit_rules(source: &str) -> Vec<&'static str> {
    rust_hits(&[("probe.rs".to_string(), source.to_string())])
        .into_iter()
        .map(|(_, hit)| hit.rule)
        .collect()
}

fn assert_caught(source: &str) {
    assert!(
        !hit_rules(source).is_empty(),
        "lint missed this production code:\n{source}"
    );
}

fn assert_clean(source: &str) {
    let rules = hit_rules(source);
    assert!(
        rules.is_empty(),
        "lint flagged a lookalike ({rules:?}):\n{source}"
    );
}

#[test]
fn benchmark_sizes_are_caught_in_constants_whatever_the_name() {
    for size in BENCHMARK_SIZES {
        assert_caught(&format!("const SHORTEST_PATH_FROM: usize = {size};"));
        assert_caught(&format!("pub(crate) const CROSSOVER: u32 = {size};"));
        assert_caught(&format!("static LIMIT: i64 = {size};"));
        assert_caught(&format!("const WIDTH: usize =\n    {size};"));
        assert_caught(&format!("const SIZES: [usize; 2] = [5, {size}];"));
        assert_caught(&format!("const HEX: usize = {size:#x};"));
        assert_caught(&format!("const UNDERSCORED: usize = {size}usize;"));
    }
    assert_caught("const FIXED_ATOMS: usize = 6;");
    assert_caught("const ACTIVE_VOLUME_MIN: usize = 8;");
    assert_caught("const MINIMUM_PACKING_ATOMS: u8 = 13;");
    assert_clean("const BUDGET: usize = 1000;");
    assert_clean("const TOLERANCE: f64 = 1e-13;");
    assert_clean("const SEED: u64 = 0x9e37_79b9;");
}

#[test]
fn benchmark_sizes_are_caught_in_comparisons_ranges_and_match_arms() {
    for size in BENCHMARK_SIZES {
        assert_caught(&format!("if n_points == {size} {{ special(); }}"));
        assert_caught(&format!("if {size} == n {{ special(); }}"));
        assert_caught(&format!("if x.len() >= {size} {{ special(); }}"));
        assert_caught(&format!(
            "if (coordinates.len() / 3) < {size} {{ special(); }}"
        ));
        assert_caught(&format!("let wide = width >= {size};"));
        assert_caught(&format!("if n_points < count.min({size}) {{ special(); }}"));
        assert_caught(&format!("for index in 0..{size} {{ visit(index); }}"));
        assert_caught(&format!("let range = 0..={size};"));
        assert_caught(&format!("let natoms = {size};"));
        assert_caught(&format!("let scratch = vec![0usize; {size}];"));
        assert_caught(&format!("let score = {size}.0;"));
        assert_caught(&format!("return {size};"));
        assert_caught(&format!("let n_atoms: usize = {size};"));
        assert_caught(&format!(
            "let config = Config {{ n_points: {size}, ..base }};"
        ));
        assert_caught(&format!(
            "let step = match n {{\n    {size} => 0.5,\n    _ => 0.3,\n}};"
        ));
        assert_caught(&format!(
            "let step = match count {{\n    1 | {size} => 0.5,\n    _ => 0.3,\n}};"
        ));
        assert_caught(&format!(
            "let e = match (rho, n_points) {{\n    (6, {size}) => -1.0,\n    _ => 0.0,\n}};"
        ));
        assert_caught(&format!(
            "if n_points\n    < {size}\n{{\n    special();\n}}"
        ));
    }
    assert_caught(
        "let step = match natoms {\n    0..=49 => 1,\n    50..=79 => 2,\n    _ => 3,\n};",
    );
    assert_caught("if atoms < 6 { return None; }");
    assert_caught("if groups[0].len() >= 4 { rigid(); }");
    assert_clean("if index + 16 <= n { unroll(); }");
    assert_clean("if moved.len() * 8 <= n { sparse(); }");
    assert_clean("if samples < limit { collect_more(); }");
    assert_clean("let mut state = seed; state ^= state << 13;");
    assert_clean("let quartile = values[(3 * values.len()) / 4]; let q75 = quartile;");
    assert_clean("let capacity = 13 + 8 * 3 * state.len();");
    assert_clean("let windows = (75, 50, 25);");
    assert_clean("for index in 0..3 { visit(index); }");
}

#[test]
fn size_keyed_tables_and_constants_that_read_them_are_caught() {
    assert_caught("const ROWS: &[(usize, f64)] = &[(49, 1.0), (79, 2.0)];");
    assert_caught(
        "const ROWS: &[(usize, Row)] = &[\n    (\n        49,\n        Row { moves: 10 },\n    ),\n];",
    );
    assert_clean("const ROWS: &[(usize, f64)] = &[(1, 1.0), (2, 2.0)];");
    let reads = rust_hits(&[
        (
            "declare.rs".to_string(),
            "pub const SWITCH_OVER: usize = 98;".to_string(),
        ),
        (
            "reader.rs".to_string(),
            "fn pick(m: usize) -> bool { m >= crate::declare::SWITCH_OVER }".to_string(),
        ),
    ]);
    assert!(
        reads
            .iter()
            .any(|(path, hit)| path == "reader.rs" && hit.rule == "reads a size constant"),
        "a reader of a size constant escaped"
    );
}

#[test]
fn species_tables_and_atomic_number_branches_are_caught() {
    assert_caught(
        "fn radius(z: u32) -> f64 { match z {\n    1 => 0.31,\n    29 => 1.32,\n    _ => 0.0,\n} }",
    );
    let table: String = (0..60).map(|index| format!("{index}.5, ")).collect();
    assert_caught(&format!("const RADII: [f64; 60] = [{table}];"));
    assert_clean("const AXES: [f64; 3] = [1.0, 0.0, 0.0];");
}

#[test]
fn motif_spellings_are_caught() {
    for spelling in [
        "let fivefold_score = 1.0;",
        "let FiveFold = 1;",
        "fn five_fold_count() {}",
        "struct IcosahedralShell;",
        "let is_icosahedral = true;",
        "let ico = shells();",
        "fn decahedral_cut() {}",
        "let pentagonal = true;",
        "let pentagons = ring_count();",
        "mod mackay_shells {}",
        "let hcp_count = 0;",
        "let fcc = 0;",
        "let bcc = 0;",
        "let template = Template::FaceCentredCubic;",
        "let group = d5h_ops();",
        "let group = cyclic_about_z(5);",
        "let lj38 = start();",
        "fn run_lj_55() {}",
        "let basin = known_ico55();",
        "let label = \"fivefold\";",
        "let system = \"lj75\";",
    ] {
        assert_caught(spelling);
    }
    assert_clean("let icon = 1; let specific = 2; let fractal = 3;");
}

#[test]
fn comments_and_test_only_items_are_not_scanned_but_code_after_them_is() {
    assert_clean("// if n == 38 { fivefold(); }");
    assert_clean("/// const ATOMS: usize = 55;\nfn documented() {}");
    assert_clean("/* if n == 38 {\n    icosahedral();\n} */\nfn commented() {}");
    assert_clean("#[cfg(test)]\nconst STALL_POINTS: usize = 13;");
    assert_clean("#[cfg(test)]\nfn fixture() {\n    if n == 38 {}\n}");
    assert_clean(
        "#[cfg(test)]\nmod fixtures {\n    const A: usize = 55;\n    fn f() { let s = \"}\"; let c = '{'; }\n}",
    );
    assert_clean("#[cfg(all(test, feature = \"x\"))]\nmod tests {\n    const A: usize = 55;\n}");
    assert_clean("#[test]\nfn single() { assert_eq!(n, 38); }");
    assert_caught(
        "#[cfg(test)]\nmod fixtures {\n    const A: usize = 55;\n}\nconst REAL_ATOMS: usize = 55;",
    );
    assert_caught("#[cfg(test)]\nmod tests {\n}\nfn live() { if n == 38 {} }");
    assert_caught("#[cfg(test)]\nconst STALL_POINTS: usize = 13;\nconst LIVE: usize = 98;");
    assert_caught("#[cfg(not(test))]\nfn live() { if n == 38 {} }");
    assert_caught("fn live() {\n    let message = \"//\"; if n == 38 {}\n}");
    assert_caught("fn live() {\n    let c = '\"'; if n == 38 {}\n}");
}

#[test]
fn python_defaults_and_motifs_are_caught() {
    for source in [
        "def run(n=38):\n    pass",
        "def run(binary, n: int = 38) -> None:\n    pass",
        "def run(\n    binary,\n    natoms: int = 55,\n):\n    pass",
        "if n == 98:\n    pass",
        "n_atoms = 104",
        "SIZES = {38: 400, 75: 4000}",
        "x = fivefold_score(coords)",
        "kind = 'icosahedral'",
        "lj38_budget = 400_000",
    ] {
        assert!(
            !python_hits(source, true).is_empty(),
            "python library lint missed:\n{source}"
        );
    }
    for source in [
        "def run(n=38):\n    pass",
        "def run(binary, n: int = 75) -> None:\n    pass",
        "if n == 98:\n    pass",
        "N = 104",
    ] {
        assert!(
            !python_hits(source, false).is_empty(),
            "python harness lint missed:\n{source}"
        );
    }
    for source in [
        "def run(n=3):\n    pass",
        "\"\"\"n = 38 in a docstring\"\"\"\nx = 1",
        "# n = 38\nx = 1",
        "x = rng.normal(size=3)",
        "tolerance = 1e-13",
    ] {
        assert!(
            python_hits(source, true).is_empty() && python_hits(source, false).is_empty(),
            "python lint flagged a lookalike:\n{source}"
        );
    }
}

fn probe_hit(text: &str) -> (String, Hit) {
    (
        "probe.rs".to_string(),
        Hit {
            line: 1,
            rule: "probe",
            text: text.to_string(),
        },
    )
}

#[test]
fn an_allowlist_entry_pins_its_statements_exactly() {
    let original = ["if n == 38 { fivefold(); }"];
    let pinned = [Allow {
        path: "probe.rs",
        needle: "fivefold",
        count: 1,
        digest: fnv(&original),
        reason: "probe entry for the audit self-test",
    }];
    let hits = [probe_hit(original[0])];
    let (unreviewed, stale) = audit(&hits, &pinned);
    assert!(unreviewed.is_empty() && stale.is_empty());

    let edited = [probe_hit("if n == 55 { fivefold(); }")];
    let (_, stale) = audit(&edited, &pinned);
    assert_eq!(stale.len(), 1, "an edited statement must re-open review");

    let added = [probe_hit(original[0]), probe_hit("let fivefold_two = 2;")];
    let (_, stale) = audit(&added, &pinned);
    assert_eq!(stale.len(), 1, "a second statement must re-open review");

    let (unreviewed, _) = audit(&[probe_hit("let icosahedral = 1;")], &pinned);
    assert_eq!(unreviewed.len(), 1, "an unclaimed hit must fail");

    let (_, stale) = audit(&[], &pinned);
    assert_eq!(stale.len(), 1, "an entry that no longer matches is stale");
}

#[test]
fn allowlist_entries_are_specific_and_point_at_live_files() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (name, allow) in [("rust", ALLOW), ("python", PYTHON_ALLOW)] {
        for (index, entry) in allow.iter().enumerate() {
            assert!(
                root.join(entry.path).is_file(),
                "{name} allowlist path {} does not exist",
                entry.path
            );
            assert!(
                entry.count >= 1,
                "{} `{}` pins no statements",
                entry.path,
                entry.needle
            );
            assert!(
                entry.digest != 0,
                "{} `{}` has no digest",
                entry.path,
                entry.needle
            );
            assert!(
                entry.needle.trim().len() >= 3,
                "{} needle `{}` is too short to be specific",
                entry.path,
                entry.needle
            );
            assert!(
                entry.reason.split_whitespace().count() >= 4,
                "{} `{}` needs a real reason",
                entry.path,
                entry.needle
            );
            for other in &allow[index + 1..] {
                assert!(
                    !(other.path == entry.path && other.needle.eq_ignore_ascii_case(entry.needle)),
                    "duplicate allowlist entry {} `{}`",
                    entry.path,
                    entry.needle
                );
            }
        }
    }
}
