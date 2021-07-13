from typing import Any
from eve import codegen
from eve.codegen import FormatTemplate as as_fmt, MakoTemplate as as_mako
from unstructured.ir import OffsetLiteral
from unstructured.backends import backend


class EmbeddedDSL(codegen.TemplatedGenerator):
    Sym = as_fmt("{id}")
    SymRef = as_fmt("{id}")
    IntLiteral = as_fmt("{value}")
    OffsetLiteral = as_fmt("{value}")
    StringLiteral = as_fmt("{value}")
    FunCall = as_fmt("{fun}({','.join(args)})")
    Lambda = as_mako("lambda ${','.join(params)}: return ${expr}")  # TODO capture
    StencilClosure = as_mako(
        "closure(${domain}, ${stencil}, [${','.join(outputs)}], [${','.join(inputs)}])"
    )
    FencilDefinition = as_mako(
        """
@fendef
def ${id}(${','.join(params)}):
    ${'\\n'.join(closures)})
    """
    )
    FunctionDefinition = as_mako(
        """
@fundef
def ${id}(${','.join(params)}):
    return ${expr}
    """
    )
    Program = as_fmt("{''.join(function_definitions)} {''.join(fencil_definitions)}")


backend.register_backend("embedded", EmbeddedDSL)
