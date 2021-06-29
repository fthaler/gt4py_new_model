from functools import total_ordering
from typing import Any, Dict, List, Optional, Set, Type
from eve import Node
from eve.traits import SymbolTableTrait
from eve.type_definitions import IntEnum, SymbolName, SymbolRef
import enum

from eve.typingx import RootValidatorType, RootValidatorValuesType
from eve.visitors import NodeVisitor
import pydantic
from devtools import debug


def validate_symbol_refs() -> RootValidatorType:
    """Validate that symbol refs are found in a symbol table valid at the current scope."""

    def _impl(
        cls: Type[pydantic.BaseModel], values: RootValidatorValuesType
    ) -> RootValidatorValuesType:
        class SymtableValidator(NodeVisitor):
            def __init__(self) -> None:
                self.missing_symbols: List[str] = []

            def visit_Node(
                self, node: Node, *, symtable: Dict[str, Any], **kwargs: Any
            ) -> None:
                for name, metadata in node.__node_children__.items():
                    if isinstance(metadata["definition"].type_, type) and issubclass(
                        metadata["definition"].type_, SymbolRef
                    ):
                        if getattr(node, name) and getattr(node, name) not in symtable:
                            self.missing_symbols.append(getattr(node, name))

                if isinstance(node, SymbolTableTrait):
                    symtable = {**symtable, **node.symtable_}
                self.generic_visit(node, symtable=symtable, **kwargs)

            @classmethod
            def apply(cls, node: Node, *, symtable: Dict[str, Any]) -> List[str]:
                instance = cls()
                instance.visit(node, symtable=symtable)
                return instance.missing_symbols

        missing_symbols = []
        for v in values.values():
            missing_symbols.extend(
                SymtableValidator.apply(v, symtable=values["symtable_"])
            )

        if len(missing_symbols) > 0:
            raise ValueError("Symbols {} not found.".format(missing_symbols))

        return values

    return pydantic.root_validator(allow_reuse=True, skip_on_failure=True)(_impl)


def get_builtins():
    builtins = ["lift", "deref", "shift", "scan", "add", "mul"]
    return set(SymbolName(name) for name in builtins)


@enum.unique
class DataType(IntEnum):
    """Data type identifier."""

    # IDs from dawn
    INVALID = 0
    AUTO = 1
    BOOLEAN = 2
    INT32 = 3
    FLOAT32 = 4
    FLOAT64 = 5
    UINT32 = 6


class Expr(Node):
    ...


class Symbol(Expr):
    name: SymbolRef


class ValueLiteral(Expr):
    value: str
    dtype: DataType


class OffsetLiteral(Expr):
    name: SymbolRef
    index: Optional[int]  # for convenience


class OffsetsLiteral(Expr):  # if we want that...
    literals: List[OffsetLiteral]


class FunctionCall(Expr):
    name: SymbolRef
    arguments: List[Expr]


class FunctionDefinition(Node):
    name: SymbolName
    parameters: List[SymbolName]
    return_expr: Expr  # or, if we have `let`, a list of `let` expr + return?


class StencilClosure(Node):
    stencil: SymbolRef
    input_fields: List[SymbolRef]
    output_fields: List[SymbolRef]
    domain: SymbolRef  # is this enough?


# class OffsetProvider(Node):
#     ...


# class CartesianOffset(OffsetProvider):
#     ...


# class OffsetGroup(OffsetProvider):
#     offsets = List[OffsetProvider]


# class OffsetMapping(Node):
#     offset_literal: SymbolName
#     offset_provider: OffsetProvider


class FencilDefinition(Node):
    name: SymbolName
    parameters: List[SymbolName]
    # offset_mapping: List[OffsetMapping]
    stencil_closures: List[StencilClosure]


class Program(Node):
    fencil_definitions: List[FencilDefinition]
    function_definitions: List[FunctionDefinition]
    _builtin_functions: List[SymbolName] = get_builtins()

    # _validate_symbol_refs = validate_symbol_refs()


# Example
# class I:
#     ...


# class J:
#     ...


# @stencil
# def lap(inp):
#     return -4.0 * deref(inp) + (
#         (deref(shift(I(-1), inp)) + deref(shift(I(-1), inp)))
#         + (deref(shift(J(-1), inp)) + deref(shift(J(-1), inp)))()
#     )


# @fencil({I: ..., J: ...})
# def lap_fencil(inp, out, dom):
#     closure(stencil=lap, domain=dom, ins=[inp], outs=[out])

# we will generate something like
# class I; // tags
# class J;

# template<class T0>
# void lap(T0 inp) {
#     return /*... + */ deref(shift(I, 1)(inp)) /* ... */
#     // ...
# }

# template<class T0, class T1, class T2>
# void lap_fencil(T0 inp, T1 out, T2 dom) {
#     apply_stencil(lap, dom)(inp, out)
# }

# and bindings:
# here we need to know the kind of OffsetProvider

left = FunctionCall(
    name="deref",
    arguments=[
        FunctionCall(
            name="shift",
            arguments=[OffsetLiteral(name="I", index=-1), Symbol(name="inp")],
        )
    ],
)
right = FunctionCall(
    name="deref",
    arguments=[
        FunctionCall(
            name="shift",
            arguments=[OffsetLiteral(name="I", index=1), Symbol(name="inp")],
        )
    ],
)
top = FunctionCall(
    name="deref",
    arguments=[
        FunctionCall(
            name="shift",
            arguments=[OffsetLiteral(name="J", index=-1), Symbol(name="inp")],
        )
    ],
)
bottom = FunctionCall(
    name="deref",
    arguments=[
        FunctionCall(
            name="shift",
            arguments=[OffsetLiteral(name="J", index=1), Symbol(name="inp")],
        )
    ],
)

neigh_sum = FunctionCall(
    name="add",
    arguments=[
        FunctionCall(name="add", arguments=[left, right]),
        FunctionCall(name="add", arguments=[top, bottom]),
    ],
)
center = FunctionCall(
    name="mul",
    arguments=[
        ValueLiteral(value="-4.0", dtype=DataType.FLOAT64),
        FunctionCall(name="deref", arguments=[Symbol(name="inp2")]),
    ],
)
lap_expr = FunctionCall(name="add", arguments=[center, neigh_sum])
stencil = FunctionDefinition(name="lap", parameters=["inp"], return_expr=lap_expr)
fencil = FencilDefinition(
    name="lap_fencil",
    parameters=["inp", "out", "dom"],
    stencil_closures=[
        StencilClosure(
            stencil="lap", domain="dom", input_fields=["inp"], output_fields=["out"]
        )
    ],
)
program = Program(fencil_definitions=[fencil], function_definitions=[stencil])

debug(program)
