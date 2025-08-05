import ast
from sympy import sympify, simplify, factor, expand

class MathOptimizer:
    """
    Traverses an AST and applies a pipeline of SymPy transforms to every BinOp:
      1) simplify()    → constant-folding & basic algebraic simplify
      2) factor()      → common-subexpression elimination
      3) expand() [opt]→ optional polynomial expansion
    """

    def __init__(self, expand_polynomials: bool = False):
        self.expand_polynomials = expand_polynomials

    def optimize_node(self, node: ast.BinOp) -> ast.AST:
        # 1) turn AST node back into source
        expr_str = ast.unparse(node)

        # 2) parse into SymPy
        sym_expr = sympify(expr_str)

        # 3) transforms
        folded  = simplify(sym_expr)
        factored = factor(folded)
        final = expand(factored) if self.expand_polynomials else factored

        # 4) back to AST
        return ast.parse(str(final)).body[0].value

    def optimize_tree(self, tree: ast.AST) -> ast.AST:
        """
        Walk the AST, replace every BinOp with its optimized version.
        """
        class Transformer(ast.NodeTransformer):
            def __init__(self, optimizer):
                self.optimizer = optimizer

            def visit_BinOp(self, node):
                node = self.generic_visit(node)
                return self.optimizer.optimize_node(node)

        return Transformer(self).visit(tree)
