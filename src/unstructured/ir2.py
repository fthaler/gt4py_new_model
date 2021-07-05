from typing import Any, List, Union
from eve import Node
from eve import codegen
from eve.type_definitions import SymbolName, SymbolRef
from eve.traits import SymbolTableTrait
from eve.codegen import TemplatedGenerator
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from devtools import debug
from yasi import indent_code
from unstructured.sym_validation import validate_symbol_refs
import inspect


_fundefs = []  # something like this or collect only reachable functions


class Sym(Node):  # helper
    id: SymbolName


class Expr(Node):
    def __add__(self, other):
        return FunCall(name=SymRef(id="plus"), args=[self, other])

    def __radd__(self, other):
        return FunCall(name=SymRef(id="plus"), args=[make_node(other), self])

    def __sub__(self, other):
        return FunCall(name=SymRef(id="minus"), args=[self, other])

    def __call__(self, *args):
        return FunCall(name=self, args=[*args])


class IntLiteral(Expr):
    value: int


class StringLiteral(Expr):
    value: str


class OffsetLiteral(Expr):
    value: Union[int, str]


class SymRef(Expr):
    id: SymbolRef


class Lambda(Expr, SymbolTableTrait):
    params: List[Sym]
    expr: Expr


class FunCall(Expr):
    name: Expr
    args: List[Expr]


class FunctionDefinition(Node, SymbolTableTrait):
    id: SymbolName
    params: List[Sym]
    expr: Expr

    def __call__(self, *args):
        return FunCall(name=SymRef(id=str(self.id)), args=[*args])


class Setq(Node):
    id: SymbolName
    expr: Expr


class StencilClosure(Node):
    domain: Expr  # CartesianDomain
    stencil: Expr  # SymbolRef[Stencil]
    outputs: List[SymRef]  # List[SymbolRef[Field]]
    inputs: List[SymRef]


class FencilDefinition(Node, SymbolTableTrait):
    id: SymbolName
    params: List[Sym]
    closures: List[StencilClosure]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not len(args) == len(self.params):
            raise RuntimeError("Incorrect number of arguments")

        prog = Program(
            function_definitions=_fundefs, fencil_definitions=[self], setqs=[]
        )
        if "backend" in kwargs:
            if kwargs["backend"] == "lisp":
                print(ToLispLike.apply(prog))
            elif kwargs["backend"] == "c++":
                print(ToyCpp.apply(prog))


class Program(Node, SymbolTableTrait):
    function_definitions: List[FunctionDefinition]
    fencil_definitions: List[FencilDefinition]
    setqs: List[Setq]

    builtin_functions = list(
        Sym(id=name)
        for name in [
            "cartesian",
            "compose",
            "lift",
            "deref",
            "shift",
            "scan",
            "plus",
            "minus",
        ]
    )
    _validate_symbol_refs = validate_symbol_refs()


class ToLispLike(TemplatedGenerator):
    Sym = as_fmt("{id}")
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
    Sym = as_fmt("{id}")
    SymRef = as_fmt("{id}")
    IntLiteral = as_fmt("{value}")

    def visit_OffsetLiteral(self, node: OffsetLiteral, **kwargs):
        return node.value if isinstance(node.value, str) else f"{node.value}_c"

    StringLiteral = as_fmt("{value}")
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


def _s(id):
    return SymRef(id=id)


def _f(fun, *args):
    if isinstance(fun, str):
        fun = _s(fun)
    return FunCall(name=fun, args=[*args])


def deref(arg):
    return _f("deref", arg)


def shift(*offsets):
    offsets = tuple(
        OffsetLiteral(value=o) if isinstance(o, str) or isinstance(o, int) else o
        for o in offsets
    )
    return _f("shift", *offsets)


def lift(sten):
    return _f("lift", sten)


def make_node(o):
    if isinstance(o, Node):
        return o
    if callable(o):
        if o.__name__ == "<lambda>":
            return lambdadef(o)
        if o.__code__.co_flags & inspect.CO_NESTED:
            return lambdadef(o)
    if isinstance(o, int):
        return IntLiteral(value=o)
    raise NotImplementedError(f"Cannot handle {o}")


def fundef(fun, *, is_lambda=False):
    body = fun(*list(_s(param) for param in inspect.signature(fun).parameters.keys()))
    body = make_node(body)
    if is_lambda:
        return Lambda(
            params=list(
                Sym(id=param) for param in inspect.signature(fun).parameters.keys()
            ),
            expr=body,
        )
    else:
        res = FunctionDefinition(
            id=fun.__name__,
            params=list(
                Sym(id=param) for param in inspect.signature(fun).parameters.keys()
            ),
            expr=body,
        )
        _fundefs.append(res)
        return res


def lambdadef(fun):
    return fundef(fun, is_lambda=True)


def compose(*args):
    return _f("compose", *args)


def cartesian(*args):
    return _f("cartesian", *args)


def apply_stencil(domain, stencil, outputs, inputs):
    return StencilClosure(
        domain=domain,
        stencil=SymRef(id=str(stencil.id)),
        outputs=outputs,
        inputs=inputs,
    )


def closure(*args):
    return list(arg for arg in args)


def fendef(fun):
    res = fun(*list(_s(param) for param in inspect.signature(fun).parameters.keys()))

    return FencilDefinition(
        id=fun.__name__,
        params=list(
            Sym(id=param) for param in inspect.signature(fun).parameters.keys()
        ),
        closures=res,
    )


# user code ===============
@fundef
def ldif(d):
    return lambda inp: deref(shift(d, -1)(inp)) - deref(inp)


@fundef
def rdif(d):
    return compose(ldif(d), shift(d, 1))


@fundef
def dif2(d):
    return compose(ldif(d), lift(rdif(d)))


i = OffsetLiteral(value="i")
j = OffsetLiteral(value="j")


@fundef
def lap(inp):
    return dif2(i)(inp) + dif2(j)(inp)


@fendef
def testee(xs, xe, ys, ye, z, output, input):
    return closure(apply_stencil(cartesian(xs, xe, ys, ye, z), lap, [output], [input]))


# my_fencil = FencilDefinition(
#     id="testee",
#     params=[
#         Sym(id="xs"),
#         Sym(id="xe"),
#         Sym(id="ys"),
#         Sym(id="ye"),
#         Sym(id="z"),
#         Sym(id="output"),
#         Sym(id="input"),
#     ],
#     closures=[
#         StencilClosure(
#             domain=FunCall(
#                 name=SymRef(id="cartesian"),
#                 args=[
#                     SymRef(id="xs"),
#                     SymRef(id="xe"),
#                     SymRef(id="ys"),
#                     SymRef(id="ye"),
#                     IntLiteral(value=0),
#                     SymRef(id="z"),
#                 ],
#             ),
#             stencil=SymRef(id="lap"),
#             outputs=[SymRef(id="output")],
#             inputs=[SymRef(id="input")],
#         )
#     ],
# )

# prog = Program(
#     function_definitions=[ldif, rdif, dif2, lap],
#     fencil_definitions=[testee],
#     setqs=[],
# )
# debug(prog)

# print(ToLispLike.apply(prog))

testee(*([None] * 7), backend="lisp")
testee(*([None] * 7), backend="c++")

# print(ToyCpp.apply(prog))
