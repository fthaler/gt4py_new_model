import functools
from typing import Any
from eve import Node
from devtools import debug
import inspect
import unstructured.builtins
import unstructured.runtime
from unstructured.ir import (
    Expr,
    FencilDefinition,
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


def patch_Expr():
    @monkeypatch_method(Expr)
    def __add__(self, other):
        return FunCall(fun=SymRef(id="plus"), args=[self, other])

    @monkeypatch_method(Expr)
    def __radd__(self, other):
        return make_node(other) + self

    @monkeypatch_method(Expr)
    def __sub__(self, other):
        return FunCall(fun=SymRef(id="minus"), args=[self, other])

    @monkeypatch_method(Expr)
    def __call__(self, *args):
        return FunCall(fun=self, args=[*args])


patch_Expr()


def patch_FunctionDefinition():
    @monkeypatch_method(FunctionDefinition)
    def __call__(self, *args):
        return FunCall(fun=SymRef(id=str(self.id)), args=[*args])


patch_FunctionDefinition()


# def patch_FencilDefinition():
#     @monkeypatch_method(FencilDefinition)
#     def __call__(self, *args: Any, **kwargs: Any) -> Any:
#         if not len(args) == len(self.params):
#             raise RuntimeError("Incorrect number of arguments")

#         prog = Program(
#             function_definitions=Tracer.fundefs, fencil_definitions=[self], setqs=[]
#         )
#         if "backend" in kwargs:
#             if kwargs["backend"] in backend._BACKENDS:
#                 b = backend.get_backend(kwargs["backend"])
#                 print(b.apply(prog))
#             else:
#                 raise RuntimeError(f"Backend {kwargs['backend']} does not exist.")


# patch_FencilDefinition()


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

    args = tuple(arg if isinstance(arg, Node) else make_node(arg) for arg in args)
    return FunCall(fun=fun, args=[*args])


# builtins

# # add all builtins
# for builtin in ["deref", "lift", "compose", "cartesian"]:
#     current_module = __import__(__name__)
#     setattr(current_module, builtin, lambda *args: _f(builtin, *args))


def deref(arg):
    return _f("deref", arg)


def lift(sten):
    return _f("lift", sten)


def compose(*args):
    return _f("compose", *args)


def cartesian(*args):
    return _f("cartesian", *args)


# runtime functions


def offset(value):
    return OffsetLiteral(value=value)


# shift promotes its arguments to literals, therefore special
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
    if isinstance(o, int):
        return IntLiteral(value=o)
    raise NotImplementedError(f"Cannot handle {o}")


def fundef(fun, *, is_lambda=False):
    if is_lambda:
        body = fun(
            *list(_s(param) for param in inspect.signature(fun).parameters.keys())
        )
        body = make_node(body)
        return Lambda(
            params=list(
                Sym(id=param) for param in inspect.signature(fun).parameters.keys()
            ),
            expr=body,
        )

    @functools.wraps(fun)
    def _dispatcher(*args):
        if not all(isinstance(arg, Node) for arg in args) and not is_lambda:
            return fun(*args)
        else:
            body = fun(
                *list(_s(param) for param in inspect.signature(fun).parameters.keys())
            )
            # debug(body)
            body = make_node(body)
            # debug(body)
            # if is_lambda:
            #     return Lambda(
            #         params=list(
            #             Sym(id=param)
            #             for param in inspect.signature(fun).parameters.keys()
            #         ),
            #         expr=body,
            #     )
            # else:
            res = FunctionDefinition(
                id=fun.__name__,
                params=list(
                    Sym(id=param) for param in inspect.signature(fun).parameters.keys()
                ),
                expr=body,
            )
            Tracer.add_fundef(res)
            return res(*args)

    return _dispatcher


unstructured.runtime._fundef_impl = fundef


def lambdadef(fun):
    return fundef(fun, is_lambda=True)


class Tracer:
    is_tracing = False
    fundefs = []

    @staticmethod
    def _enable_tracing():
        unstructured.builtins._deref_impl = deref
        unstructured.builtins._lift_impl = lift
        unstructured.builtins._shift_impl = shift

        unstructured.runtime._closures_impl = closures

    @staticmethod
    def _disable_tracing():
        unstructured.builtins._deref_impl = unstructured.builtins.default_impl
        unstructured.builtins._lift_impl = unstructured.builtins.default_impl
        unstructured.builtins._shift_impl = unstructured.builtins.default_impl

        unstructured.runtime._closures_impl = lambda *args: ...

    @classmethod
    def add_fundef(cls, fun):
        if not fun in cls.fundefs:
            cls.fundefs.append(fun)

    def __enter__(self):
        self.is_tracing = True
        self._enable_tracing()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._disable_tracing()
        self.fundefs = []
        self.is_tracing = False


def apply_stencil(domain, stencil, outputs, inputs):
    stencil(*list(_s(param) for param in inspect.signature(stencil).parameters.keys()))
    return StencilClosure(
        domain=domain,
        stencil=SymRef(id=str(stencil.__name__)),
        outputs=outputs,
        inputs=inputs,
    )


def closures(*args):
    print("closures")
    return list(arg for arg in args)


def fendef(fun):
    @functools.wraps(fun)
    def _dispatcher(*args, **kwargs):
        if "backend" in kwargs:
            with Tracer() as _:
                res = fun(
                    *list(
                        _s(param) for param in inspect.signature(fun).parameters.keys()
                    )
                )

                fencil = FencilDefinition(
                    id=fun.__name__,
                    params=list(
                        Sym(id=param)
                        for param in inspect.signature(fun).parameters.keys()
                    ),
                    closures=res,
                )
            execute_program(fencil, *args, **kwargs)
        else:
            return fun(*args)

    return _dispatcher


unstructured.runtime._fendef_impl = fendef
