import typer
import ast
import difflib
from pathlib import Path
from parser import MathExtractor
from optimizer import MathOptimizer
from rich.console import Console
from rich.panel import Panel
from sympy import sympify, diff as sym_diff, integrate as sym_int

app = typer.Typer()
console = Console()

def parse_line_set(spec: str) -> set[int]:
    """Turn '2,5-7' into {2,5,6,7}."""
    lines = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lines.update(range(int(a), int(b) + 1))
        else:
            lines.add(int(part))
    return lines

def process_file(
    path: Path,
    optimizer: MathOptimizer,
    inplace: bool,
    show_diff: bool,
    diff_vars: set[str],
    int_vars: set[str],
    diff_lines: set[int],
):
    """
    1) Read source
    2) Optionally differentiate/integrate filtered assigns
    3) Run math-optimization pipeline
    4) Either overwrite or show unified diff/raw code
    """
    src = path.read_text()
    tree = ast.parse(src)

    # 2) Symbolic diff/int pass
    if diff_vars or int_vars or diff_lines:
        class CalcTransformer(ast.NodeTransformer):
            def visit_Assign(self, node: ast.Assign):
                # Only single-target name assignments
                target = node.targets[0]
                if isinstance(target, ast.Name) and isinstance(node.value, ast.AST):
                    varname = target.id
                    lineno = node.lineno
                    do_diff = varname in diff_vars and (not diff_lines or lineno in diff_lines)
                    do_int  = varname in int_vars  and (not diff_lines or lineno in diff_lines)

                    if do_diff:
                        expr = sympify(ast.unparse(node.value))
                        der  = sym_diff(expr, varname)
                        node.value = ast.parse(str(der)).body[0].value

                    if do_int:
                        expr = sympify(ast.unparse(node.value))
                        ant  = sym_int(expr, varname)
                        node.value = ast.parse(str(ant)).body[0].value

                return node

        tree = CalcTransformer().visit(tree)

    # 3) Math optimization
    full_src = ast.unparse(tree)
    extractor = MathExtractor()
    tree2, _ = extractor.extract(full_src)
    optimized = ast.unparse(optimizer.optimize_tree(tree2))

    # 4) Output
    if inplace:
        path.write_text(optimized)
        console.print(f"✅ Updated: {path}")
        return

    if show_diff:
        diff_txt = "".join(difflib.unified_diff(
            full_src.splitlines(keepends=True),
            optimized.splitlines(keepends=True),
            fromfile=str(path),
            tofile="optimized",
            lineterm="",
        )) or "[italic]No changes[/italic]"
        console.print(Panel(diff_txt, title=str(path), border_style="blue"))
    else:
        console.print(Panel(optimized, title=str(path), border_style="green"))

@app.command()
def main(
    target: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=True,
        help="File or directory to process"
    ),
    expand: bool = typer.Option(
        False, "--expand/--no-expand",
        help="Enable polynomial expand pass"
    ),
    differentiate: list[str] = typer.Option(
        [], "-d", "--differentiate",
        help="Variable names to differentiate"
    ),
    integrate: list[str] = typer.Option(
        [], "-i", "--integrate",
        help="Variable names to integrate"
    ),
    diff_lines: str = typer.Option(
        "", "--diff-lines",
        help="Comma/range list of line numbers to filter diff/int"
    ),
    inplace: bool = typer.Option(
        False, "--inplace",
        help="Overwrite source files in place"
    ),
    recursive: bool = typer.Option(
        False, "--recursive",
        help="When target is a directory, recurse into subfolders"
    ),
    diff: bool = typer.Option(
        True, "--diff/--no-diff",
        help="Show unified diff instead of raw code"
    ),
):
    """
    Optimize math (and optionally differentiate/integrate) in FILE or all .py under a directory.
    """
    diff_vars = set(differentiate)
    int_vars  = set(integrate)
    lines_set = parse_line_set(diff_lines) if diff_lines else set()
    optimizer = MathOptimizer(expand_polynomials=expand)

    paths = ([target] if target.is_file()
             else sorted(target.glob("**/*.py") if recursive else target.glob("*.py")))
    for path in paths:
        process_file(path, optimizer, inplace, diff, diff_vars, int_vars, lines_set)

if __name__ == "__main__":
    app()
