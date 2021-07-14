import functools
from typing import Any
from typing_extensions import runtime
from eve import Node
from devtools import debug
import inspect
from unstructured.backend_executor import execute_program
import unstructured.builtins
import unstructured.runtime
from unstructured.ir import (
    Expr,
    FencilDefinition,
    FloatLiteral,
    FunCall,
    FunctionDefinition,
    IntLiteral,
    Lambda,
    OffsetLiteral,
    Program,
    StencilClosure,
    Sym,
    SymRef,
)
from unstructured.backends import backend


def monkeypatch_method(cls):
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func

    return decorator


def _patch_Expr():
    @monkeypatch_method(Expr)
    def __add__(self, other):
        return FunCall(fun=SymRef(id="plus"), args=[self, make_node(other)])

    @monkeypatch_method(Expr)
    def __radd__(self, other):
        return make_node(other) + self

    @monkeypatch_method(Expr)
    def __mul__(self, other):
        return FunCall(fun=SymRef(id="mul"), args=[self, make_node(other)])

    @monkeypatch_method(Expr)
    def __rmul__(self, other):
        return make_node(other) * self

    @monkeypatch_method(Expr)
    def __sub__(self, other):
        return FunCall(fun=SymRef(id="minus"), args=[self, make_node(other)])

    @monkeypatch_method(Expr)
    def __gt__(self, other):
        return FunCall(fun=SymRef(id="greater"), args=[self, make_node(other)])

    @monkeypatch_method(Expr)
    def __lt__(self, other):
        return FunCall(fun=SymRef(id="less"), args=[self, make_node(other)])

    @monkeypatch_method(Expr)
    def __call__(self, *args):
        return FunCall(fun=self, args=[*make_node(args)])


_patch_Expr()


class PatchedFunctionDefinition(FunctionDefinition):
    def __call__(self, *args):
        return FunCall(fun=SymRef(id=str(self.id)), args=[*make_node(args)])


def _s(id):
    return SymRef(id=id)


def trace_function_argument(arg):
    if isinstance(arg, unstructured.runtime.FundefDispatcher):
        make_function_definition(arg.fun)
        return _s(arg.fun.__name__)
    return arg


def _f(fun, *args):
    if isinstance(fun, str):
        fun = _s(fun)

    args = [trace_function_argument(arg) for arg in args]
    return FunCall(fun=fun, args=[*make_node(args)])


# builtins
@unstructured.builtins.deref.register("tracing")
def deref(arg):
    return _f("deref", arg)


@unstructured.builtins.lift.register("tracing")
def lift(sten):
    return _f("lift", sten)


@unstructured.builtins.compose.register("tracing")
def compose(*args):
    return _f("compose", *args)


@unstructured.builtins.cartesian.register("tracing")
def cartesian(*args):
    return _f("cartesian", *args)


@unstructured.builtins.if_.register("tracing")
def if_(*args):
    return _f("if_", *args)


# shift promotes its arguments to literals, therefore special
@unstructured.builtins.shift.register("tracing")
def shift(*offsets):
    offsets = tuple(
        OffsetLiteral(value=o) if isinstance(o, (str, int)) else o for o in offsets
    )
    return _f("shift", *offsets)


@unstructured.builtins.plus.register("tracing")
def plus(*args):
    return _f("plus", *args)


@unstructured.builtins.minus.register("tracing")
def minus(*args):
    return _f("minus", *args)


@unstructured.builtins.mul.register("tracing")
def mul(*args):
    return _f("mul", *args)


@unstructured.builtins.greater.register("tracing")
def greater(*args):
    return _f("greater", *args)


# helpers
def make_node(o):
    if isinstance(o, Node):
        return o
    if callable(o):
        if o.__name__ == "<lambda>":
            return lambdadef(o)
        if hasattr(o, "__code__") and o.__code__.co_flags & inspect.CO_NESTED:
            return lambdadef(o)
    if isinstance(o, unstructured.runtime.Offset):
        return OffsetLiteral(value=o.value)
    if isinstance(o, int):
        return IntLiteral(value=o)
    if isinstance(o, float):
        return FloatLiteral(value=o)
    if isinstance(o, tuple):
        return tuple(make_node(arg) for arg in o)
    if isinstance(o, list):
        return list(make_node(arg) for arg in o)
    if o is None:
        return None
    raise NotImplementedError(f"Cannot handle {o}")


def trace_function_call(fun):
    body = fun(*list(_s(param) for param in inspect.signature(fun).parameters.keys()))
    return make_node(body)


def lambdadef(fun):
    return Lambda(
        params=list(
            Sym(id=param) for param in inspect.signature(fun).parameters.keys()
        ),
        expr=trace_function_call(fun),
    )


def make_function_definition(fun):
    res = PatchedFunctionDefinition(
        id=fun.__name__,
        params=list(
            Sym(id=param) for param in inspect.signature(fun).parameters.keys()
        ),
        expr=trace_function_call(fun),
    )
    Tracer.add_fundef(res)
    return res


class FundefTracer:
    def __call__(self, fundef_dispatcher: unstructured.runtime.FundefDispatcher):
        def fun(*args):
            res = make_function_definition(fundef_dispatcher.fun)
            return res(*args)

        return fun

    def __bool__(self):
        return unstructured.builtins.builtin_dispatch.key == "tracing"


unstructured.runtime.FundefDispatcher.register_tracing_hook(FundefTracer())


class Tracer:
    fundefs = []
    closures = []

    @classmethod
    def add_fundef(cls, fun):
        if not fun in cls.fundefs:
            cls.fundefs.append(fun)

    @classmethod
    def add_closure(cls, closure):
        cls.closures.append(closure)

    def __enter__(self):
        unstructured.builtins.builtin_dispatch.push_key("tracing")

    def __exit__(self, exc_type, exc_value, exc_traceback):
        type(self).fundefs = []
        type(self).closures = []
        unstructured.builtins.builtin_dispatch.pop_key()


@unstructured.runtime.closure.register("tracing")
def closure(domain, stencil, outputs, inputs):
    stencil(*list(_s(param) for param in inspect.signature(stencil).parameters.keys()))
    Tracer.add_closure(
        StencilClosure(
            domain=domain,
            stencil=SymRef(id=str(stencil.__name__)),
            outputs=outputs,
            inputs=inputs,
        )
    )


def fendef_tracing(fun, *args, **kwargs):
    with Tracer() as _:
        trace_function_call(fun)

        fencil = FencilDefinition(
            id=fun.__name__,
            params=list(
                Sym(id=param) for param in inspect.signature(fun).parameters.keys()
            ),
            closures=Tracer.closures,
        )
        prog = Program(
            function_definitions=Tracer.fundefs, fencil_definitions=[fencil], setqs=[]
        )
    # after tracing is done
    execute_program(prog, *args, **kwargs)


unstructured.runtime.fendef_registry[
    lambda kwargs: "backend" in kwargs
] = fendef_tracing
