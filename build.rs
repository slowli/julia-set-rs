//! Compresses the shader source code, removing comments and formatting whitespace.
//! This is useful because shader code is included into the compilation output.

use std::{env, fs, path::Path};

fn process_line(mut line: &str) -> Option<String> {
    if let Some(comment_start) = line.find("//") {
        line = &line[..comment_start];
    }

    line = line.trim();
    if line.is_empty() {
        return None;
    }

    let parts: Vec<_> = line.split_ascii_whitespace().collect();
    let mut compact_line = String::new();
    let is_definition = parts.first() == Some(&"#define");
    let ident_char = |ch: char| ch.is_ascii_alphanumeric() || ch == '_';

    for (i, pair) in parts.windows(2).enumerate() {
        match pair {
            [first, second] => {
                compact_line.push_str(first);
                if (first.ends_with(ident_char) && second.starts_with(ident_char))
                    || (is_definition && i <= 1)
                {
                    compact_line.push(' ');
                }
            }
            _ => unreachable!(),
        }
    }
    if let Some(last_part) = parts.last() {
        compact_line.push_str(last_part);
    }

    Some(compact_line)
}

fn process_shader(input_path: &str, output_name: &str) {
    let cl_code = fs::read_to_string(input_path).expect("Cannot read shader code");
    println!("cargo:rerun-if-changed={}", input_path);

    let mut processed_code = String::new();
    for line in cl_code.lines().filter_map(process_line) {
        let last_char = processed_code.as_bytes().last().cloned();
        if line.starts_with('#') && last_char.is_some() && last_char != Some(b'\n') {
            processed_code.push('\n');
        }
        processed_code.push_str(&line);
        if line.starts_with('#') {
            processed_code.push('\n');
        }
    }
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir).join(output_name);
    fs::write(out_path, processed_code).expect("Cannot write shader code");
}

fn main() {
    process_shader("src/program.cl", "program.cl");
    process_shader("src/program.glsl", "program.glsl");
}
