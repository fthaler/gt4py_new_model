from abc import ABC
from typing import Any, List
from eve import Node
from eve import codegen
from eve.type_definitions import SymbolName, SymbolRef
from eve.codegen import TemplatedGenerator
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from devtools import debug
from yasi import indent_code


class Param(Node):  # helper
    id: SymbolName


class Expr(Node):
    ...


class IntLiteral(Expr):
    value: int


class StringLiteral(Expr):
    value: str


class OffsetLiteral(Expr):
    value: int
    # name: str
    # index: int


class FunctionDefinition(Node):
    id: SymbolName
    params: List[Param]
    expr: Expr


class SymRef(Expr):
    id: SymbolRef


class Lambda(Expr):
    params: List[Param]
    expr: Expr


class FunCall(Expr):
    name: Expr
    args: List[Expr]


class Setq(Node):
    id: SymbolName
    expr: Expr


class StencilClosure(Node):
    domain: Expr  # CartesianDomain
    stencil: Expr  # SymbolRef[Stencil]
    outputs: List[SymRef]  # List[SymbolRef[Field]]
    inputs: List[SymRef]


class FencilDefinition(Node):
    id: SymbolName
    params: List[Param]
    closures: List[StencilClosure]


class Program(Node):
    function_definitions: List[FunctionDefinition]
    fencil_definitions: List[FencilDefinition]
    setqs: List[Setq]


class ToLispLike(TemplatedGenerator):
    Param = as_fmt("{id}")
    FunCall = as_fmt("({name} {' '.join(args)})")
    IntLiteral = as_fmt("{value}")
    OffsetLiteral = as_fmt("{value}")
    StringLiteral = as_fmt("{value}")
    SymRef = as_fmt("{id}")
    Program = as_fmt(
        """
    {''.join(function_definitions)}
    {''.join(fencil_definitions)}
    {''.join(setqs)}
    """
    )
    StencilClosure = as_mako(
        """(
     :domain ${domain}
     :stencil ${stencil}
     :outputs ${' '.join(outputs)}
     :inputs ${' '.join(inputs)}
    )
    """
    )
    FencilDefinition = as_mako(
        """(defen ${id}(${' '.join(params)})
        ${''.join(closures)})
        """
    )
    FunctionDefinition = as_fmt(
        """(defun {id}({' '.join(params)})
        {expr}
        )

"""
    )
    Lambda = as_fmt(
        """(lambda ({' '.join(params)})
         {expr}
          )"""
    )

    @classmethod
    def apply(cls, root, **kwargs: Any) -> str:
        generated_code = super().apply(root, **kwargs)
        indented = indent_code(generated_code, "--dialect lisp")
        formatted_code = "".join(indented["indented_code"])
        return formatted_code


class ToyCpp(TemplatedGenerator):
    SymRef = as_fmt("{id}")
    IntLiteral = as_fmt("{value}")
    OffsetLiteral = as_fmt("{value}_c")
    StringLiteral = as_fmt("{value}")
    Param = as_fmt("{id}")
    FunCall = as_fmt("{name}({','.join(args)})")
    Lambda = as_mako(
        "[=](${','.join('auto ' + p for p in params)}){return ${expr};}"
    )  # TODO capture
    StencilClosure = as_mako(
        "closure(${domain}, ${stencil}, out(${','.join(outputs)}), ${','.join(inputs)})"
    )
    FencilDefinition = as_mako(
        """
    auto ${id} = [](${','.join('auto&& ' + p for p in params)}){
        fencil(${'\\n'.join(closures)});
    };
    """
    )
    FunctionDefinition = as_mako(
        """
    inline constexpr auto ${id} = [](${','.join('auto ' + p for p in params)}){
        return ${expr};
        };
    """
    )
    Program = as_fmt("{''.join(function_definitions)} {''.join(fencil_definitions)}")

    @classmethod
    def apply(cls, root, **kwargs: Any) -> str:
        generated_code = super().apply(root, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code


def s(id):
    return SymRef(id=id)


def f(fun, *args):
    if isinstance(fun, str):
        fun = s(fun)
    return FunCall(name=fun, args=[*args])


def deref(arg):
    return f("deref", arg)


def minus(arg0, arg1):
    return f("minus", arg0, arg1)


def shift(it, *offsets):
    return f(f("shift", *offsets), it)


ldif = FunctionDefinition(
    id="ldif",
    params=[Param(id="d")],
    expr=Lambda(
        params=[Param(id="in")],
        expr=(
            minus(
                deref(
                    shift(SymRef(id="in"), SymRef(id="d"), OffsetLiteral(value=-1))
                ),  # What's OffsetLiteral?
                deref(SymRef(id="in")),
            )
        ),
    ),
)

rdif = FunctionDefinition(
    id="rdif",
    params=[Param(id="d")],
    expr=f("compose", f("ldif", s("d")), f("shift", s("d"), OffsetLiteral(value=1))),
)

dif2 = FunctionDefinition(
    id="dif2",
    params=[Param(id="d")],
    expr=f("compose", f("ldif", s("d")), f("lift", f("rdif", s("d")))),
)

lap = FunctionDefinition(
    id="lap",
    params=[Param(id="in")],
    expr=f(
        "plus",
        f(f("dif2", StringLiteral(value="i")), s("in")),
        f(f("dif2", StringLiteral(value="j")), s("in")),
    ),
)

my_fencil = FencilDefinition(
    id="testee",
    params=[
        Param(id="xs"),
        Param(id="xe"),
        Param(id="ys"),
        Param(id="ye"),
        Param(id="z"),
        Param(id="output"),
        Param(id="input"),
    ],
    closures=[
        StencilClosure(
            domain=FunCall(
                name=SymRef(id="cartesian"),
                args=[
                    SymRef(id="xs"),
                    SymRef(id="xe"),
                    SymRef(id="ys"),
                    SymRef(id="ye"),
                    IntLiteral(value=0),
                    SymRef(id="z"),
                ],
            ),
            stencil=SymRef(id="lap"),
            outputs=[SymRef(id="output")],
            inputs=[SymRef(id="input")],
        )
    ],
)

prog = Program(
    function_definitions=[ldif, rdif, dif2, lap],
    fencil_definitions=[my_fencil],
    setqs=[],
)
debug(prog)

print(ToLispLike.apply(prog))

print(ToyCpp.apply(prog))