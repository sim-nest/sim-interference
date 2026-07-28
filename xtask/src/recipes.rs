//! Checked cargo-example harness for repository recipes.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn run() -> Result<(), String> {
    let root = std::env::current_dir().map_err(|err| format!("current dir: {err}"))?;
    let mut recipes = Vec::new();
    collect_recipe_files(&root, &mut recipes)?;
    recipes.sort();
    if recipes.is_empty() {
        return Err("check-recipes: no recipe.toml files found".to_owned());
    }

    for recipe in &recipes {
        run_recipe(&root, recipe)?;
    }
    println!("check-recipes: checked {} recipe(s)", recipes.len());
    Ok(())
}

fn run_recipe(root: &Path, recipe: &Path) -> Result<(), String> {
    let text =
        fs::read_to_string(recipe).map_err(|err| format!("read {}: {err}", recipe.display()))?;
    let harness = field(&text, "harness")?;
    if harness == "cargo-test" {
        return run_test_recipe(root, recipe, &text);
    }
    if harness != "cargo-example" {
        return Err(format!(
            "{}: unsupported recipe harness {harness:?}",
            relative_path(root, recipe)
        ));
    }
    let package = field(&text, "package")?;
    let example = field(&text, "example")?;
    let expected_name = field(&text, "expected")?;
    let expected_path = recipe
        .parent()
        .ok_or_else(|| format!("recipe has no parent: {}", recipe.display()))?
        .join(expected_name);
    let expected = fs::read_to_string(&expected_path)
        .map_err(|err| format!("read {}: {err}", expected_path.display()))?;

    let output = Command::new("cargo")
        .args(["run", "--quiet", "-p", package, "--example", example])
        .current_dir(root)
        .output()
        .map_err(|err| format!("run recipe {}: {err}", relative_path(root, recipe)))?;
    if !output.status.success() {
        return Err(format!(
            "recipe {} failed:\n{}",
            relative_path(root, recipe),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let actual = String::from_utf8(output.stdout)
        .map_err(|err| format!("recipe {} emitted non-UTF-8: {err}", recipe.display()))?;
    if actual.trim_end() != expected.trim_end() {
        return Err(format!(
            "recipe {} output mismatch\nexpected:\n{}\nactual:\n{}",
            relative_path(root, recipe),
            expected.trim_end(),
            actual.trim_end()
        ));
    }
    Ok(())
}

fn run_test_recipe(root: &Path, recipe: &Path, text: &str) -> Result<(), String> {
    let package = field(text, "package")?;
    let test = field(text, "test")?;
    let output = Command::new("cargo")
        .args(["test", "--quiet", "-p", package, test, "--", "--exact"])
        .current_dir(root)
        .output()
        .map_err(|err| format!("run recipe {}: {err}", relative_path(root, recipe)))?;
    if !output.status.success() {
        return Err(format!(
            "recipe {} failed:\n{}",
            relative_path(root, recipe),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

fn field<'a>(text: &'a str, name: &str) -> Result<&'a str, String> {
    let prefix = format!("{name} = \"");
    text.lines()
        .find_map(|line| line.strip_prefix(&prefix)?.strip_suffix('"'))
        .ok_or_else(|| format!("recipe metadata missing quoted {name}"))
}

fn collect_recipe_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    let name = path.file_name().and_then(|value| value.to_str());
    if matches!(name, Some(".git" | ".sim" | "target")) {
        return Ok(());
    }
    for entry in fs::read_dir(path).map_err(|err| format!("read {}: {err}", path.display()))? {
        let entry = entry.map_err(|err| format!("read {} entry: {err}", path.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_recipe_files(&path, files)?;
        } else if path.file_name().and_then(|value| value.to_str()) == Some("recipe.toml") {
            files.push(path);
        }
    }
    Ok(())
}

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}
