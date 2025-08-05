# File: tests/test_optimizer.py

import ast
import pytest
from parser import MathExtractor
from optimizer import MathOptimizer
from sympy import sympify, integrate as sym_int, diff as sym_diff, Symbol

# ─── 1) Simple constant-folding tests ────────────────────────────────────────────

@pytest.mark.parametrize("src, expected", [
    ("y = 4 * (2 + 3)",             "y = 20"),
    ("z = (1 + 2) * ((3 + 4) + 5)",  "z = 36"),
    ("a = 2 * x + 2 * x",           "a = 4 * x"),
    ("b = (x + 3) + (x + 3)",       "b = 2 * (x + 3)"),
    ("c = 5 - (2 + 3)",             "c = 0"),
])
def test_simple_folding(src, expected):
    tree, _ = MathExtractor().extract(src)
    out = ast.unparse(MathOptimizer().optimize_tree(tree)).strip()
    assert out == expected

def test_nested_constants():
    src = "result = 2 * (3 + (4 + 1))"
    tree, _ = MathExtractor().extract(src)
    out = ast.unparse(MathOptimizer().optimize_tree(tree)).strip()
    assert out == "result = 16"

def test_no_change_for_non_math():
    src = "print('hello')"
    tree, _ = MathExtractor().extract(src)
    out = ast.unparse(MathOptimizer().optimize_tree(tree)).strip()
    assert out == src


# ─── 2) Factor & Expand tests ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    "src, expected_no_expand, expected_expand",
    [
        ("a = (x + 2) * (x + 2)",
         "a = (x + 2) ** 2",
         "a = x ** 2 + 4 * x + 4"),
        ("b = (x - 1) * (2 * x + 2)",
         "b = 2 * (x - 1) * (x + 1)",
         "b = 2 * x ** 2 - 2"),
    ],
)
def test_factor_and_expand(src, expected_no_expand, expected_expand):
    tree1, _ = MathExtractor().extract(src)
    out1 = ast.unparse(
        MathOptimizer(expand_polynomials=False).optimize_tree(tree1)
    ).strip()
    assert out1 == expected_no_expand

    tree2, _ = MathExtractor().extract(src)
    out2 = ast.unparse(
        MathOptimizer(expand_polynomials=True).optimize_tree(tree2)
    ).strip()
    assert out2 == expected_expand


# ─── 3) Differentiation tests ───────────────────────────────────────────────────

def run_diff(src: str) -> str:
    tree = ast.parse(src)
    class DiffTransformer(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign):
            if isinstance(node.targets[0], ast.Name):
                expr = sympify(ast.unparse(node.value))
                der  = sym_diff(expr, Symbol("x"))
                node.value = ast.parse(str(der)).body[0].value
            return node

    tree = DiffTransformer().visit(tree)
    full = ast.unparse(tree)
    tree2, _ = MathExtractor().extract(full)
    return ast.unparse(MathOptimizer().optimize_tree(tree2)).strip()

@pytest.mark.parametrize("src,expected", [
    ("x = 2 * (3 + 5)", "x = 0"),
    ("y = x**3",        "y = 3 * x ** 2"),
    ("z = 5",           "z = 0"),
])
def test_simple_diff(src, expected):
    assert run_diff(src) == expected


# ─── 4) Integration tests ──────────────────────────────────────────────────────

def run_int(src: str) -> str:
    tree = ast.parse(src)
    class IntTransformer(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign):
            if isinstance(node.targets[0], ast.Name):
                expr = sympify(ast.unparse(node.value))
                ant  = sym_int(expr, Symbol("x"))
                node.value = ast.parse(str(ant)).body[0].value
            return node

    tree = IntTransformer().visit(tree)
    full = ast.unparse(tree)
    tree2, _ = MathExtractor().extract(full)
    return ast.unparse(MathOptimizer().optimize_tree(tree2)).strip()

@pytest.mark.parametrize("src,expected", [
    ("y = 6",     "y = 6 * x"),
    ("z = 2 * x", "z = x ** 2"),
    ("a = x**2",  "a = x ** 3 / 3"),
    ("b = x + 1", "b = x * (x + 2) / 2"),
])
def test_simple_integration(src, expected):
    assert run_int(src) == expected

def test_integration_then_folding():
    src = "c = 2 + 3"
    # ∫(2+3)dx = 5*x, folding keeps "5 * x"
    assert run_int(src) == "c = 5 * x"
