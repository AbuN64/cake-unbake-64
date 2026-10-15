from collections.abc import Iterator
from typing import Any, ClassVar

from .c_parser import Coord

class Node:
    attr_names: ClassVar[tuple[str, ...]]
    coord: Coord | None
    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, name: str, value: Any) -> None: ...
    def children(self) -> tuple[tuple[str, Any], ...]: ...
    def __iter__(self) -> Iterator[Any]: ...

class NodeVisitor:
    def visit(self, node: Any) -> Any: ...
    def generic_visit(self, node: Any) -> Any: ...

class Alignas(Node):
    alignment: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class ArrayDecl(Node):
    type: Any
    dim: Any
    dim_quals: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class ArrayRef(Node):
    name: Any
    subscript: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Assignment(Node):
    op: str
    lvalue: Any
    rvalue: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class BinaryOp(Node):
    op: str
    left: Any
    right: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Break(Node):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Case(Node):
    expr: Any
    stmts: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Cast(Node):
    to_type: Any
    expr: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Compound(Node):
    block_items: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class CompoundLiteral(Node):
    type: Any
    init: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Constant(Node):
    type: Any
    value: str
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Continue(Node):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Decl(Node):
    name: str | None
    quals: list[str]
    align: Any
    storage: list[str]
    funcspec: list[str]
    type: Any
    init: Any
    bitsize: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class DeclList(Node):
    decls: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Default(Node):
    stmts: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class DoWhile(Node):
    cond: Any
    stmt: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class EllipsisParam(Node):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class EmptyStatement(Node):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Enum(Node):
    name: str | None
    values: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Enumerator(Node):
    name: str | None
    value: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class EnumeratorList(Node):
    enumerators: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class ExprList(Node):
    exprs: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class FileAST(Node):
    ext: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class For(Node):
    init: Any
    cond: Any
    next: Any
    stmt: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class FuncCall(Node):
    name: Any
    args: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class FuncDecl(Node):
    args: Any
    type: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class FuncDef(Node):
    decl: Any
    param_decls: Any
    body: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Goto(Node):
    name: str
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class ID(Node):
    name: str
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class IdentifierType(Node):
    names: list[str]
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class If(Node):
    cond: Any
    iftrue: Any
    iffalse: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class InitList(Node):
    exprs: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Label(Node):
    name: str
    stmt: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class NamedInitializer(Node):
    name: Any
    expr: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class ParamList(Node):
    params: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Pragma(Node):
    string: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class PtrDecl(Node):
    quals: list[str]
    type: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Return(Node):
    expr: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class StaticAssert(Node):
    cond: Any
    message: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Struct(Node):
    name: str | None
    decls: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class StructRef(Node):
    name: Any
    type: Any
    field: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Switch(Node):
    cond: Any
    stmt: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class TernaryOp(Node):
    cond: Any
    iftrue: Any
    iffalse: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class TypeDecl(Node):
    declname: str | None
    quals: list[str]
    align: Any
    type: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Typedef(Node):
    name: str
    quals: list[str]
    storage: list[str]
    type: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Typename(Node):
    name: str | None
    quals: list[str]
    align: Any
    type: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class UnaryOp(Node):
    op: str
    expr: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Union(Node):
    name: str | None
    decls: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class While(Node):
    cond: Any
    stmt: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
