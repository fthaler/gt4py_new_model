import functools
from typing import Any
from typing_extensions import runtime
from eve import Node
from devtools import debug
import inspect
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
    def __sub__(self, other):
        return FunCall(fun=SymRef(id="minus"), args=[self, make_node(other)])

    @monkeypatch_method(Expr)
    def __call__(self, *args):
        return FunCall(fun=self, args=[*make_node(args)])


_patch_Expr()


def _patch_FunctionDefinition():
    @monkeypatch_method(FunctionDefinition)
    def __call__(self, *args):
        return FunCall(fun=SymRef(id=str(self.id)), args=[*make_node(args)])


_patch_FunctionDefinition()


def execute_program(fencil: FencilDefinition, *args, **kwargs):
    assert "backend" in kwargs
    if not len(args) == len(fencil.params):
        raise RuntimeError("Incorrect number of arguments")

    prog = Program(
        function_definitions=Tracer.fundefs, fencil_definitions=[fencil], setqs=[]
    )
    if kwargs["backend"] in backend._BACKENDS:
        b = backend.get_backend(kwargs["backend"])
        print(b.apply(prog))
    else:
        raise RuntimeError(f"Backend {kwargs['backend']} does not exist.")


def _s(id):
    return SymRef(id=id)


def _f(fun, *args):
    if isinstance(fun, str):
        fun = _s(fun)

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


# shift promotes its arguments to literals, therefore special
@unstructured.builtins.shift.register("tracing")
def shift(*offsets):
    offsets = tuple(
        OffsetLiteral(value=o) if isinstance(o, (str, int)) else o for o in offsets
    )
    return _f("shift", *offsets)


# helpers
def make_node(o):
    if isinstance(o, Node):
        return o
    if callable(o):
        if o.__name__ == "<lambda>":
            return lambdadef(o)
        if o.__code__.co_flags & inspect.CO_NESTED:
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


@unstructured.runtime.fundef.register("tracing")
def fundef(fun):
    @functools.wraps(fun)
    def _dispatcher(*args):
        if Tracer.is_tracing:
            res = FunctionDefinition(
                id=fun.__name__,
                params=list(
                    Sym(id=param) for param in inspect.signature(fun).parameters.keys()
                ),
                expr=trace_function_call(fun),
            )
            Tracer.add_fundef(res)
            return res(*args)
        else:
            return fun(*args)

    return _dispatcher


unstructured.runtime.fun_fen_def_dispatch.push_key("tracing")


# TODO Context manager from stdlib "contextlib"


class Tracer:
    is_tracing = False
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
        assert not type(self).is_tracing
        type(self).is_tracing = True
        unstructured.builtins.builtin_dispatch.push_key("tracing")

    def __exit__(self, exc_type, exc_value, exc_traceback):
        type(self).fundefs = []
        type(self).closures = []
        type(self).is_tracing = False
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


@unstructured.runtime.fendef.register("tracing")
def fendef(fun):
    @functools.wraps(fun)
    def _dispatcher(*args, **kwargs):
        if "backend" in kwargs:
            with Tracer() as _:
                trace_function_call(fun)

                fencil = FencilDefinition(
                    id=fun.__name__,
                    params=list(
                        Sym(id=param)
                        for param in inspect.signature(fun).parameters.keys()
                    ),
                    closures=Tracer.closures,
                )
                execute_program(fencil, *args, **kwargs)
        else:
            return fun(*args)

    return _dispatcher
