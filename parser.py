import ast

class MathExtractor(ast.NodeVisitor):
    """
    Walks an AST and collects BinOp nodes.
    """
    def __init__(self):
        self.nodes = []

    def visit_BinOp(self, node: ast.BinOp):
        self.nodes.append(node)
        self.generic_visit(node)

    def extract(self, source: str):
        """
        Parse Python source into an AST, record all BinOp nodes,
        and return (tree, list_of_binops).
        """
        tree = ast.parse(source)
        self.visit(tree)
        return tree, self.nodes
