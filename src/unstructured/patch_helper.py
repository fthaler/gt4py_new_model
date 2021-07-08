from typing import Any


class Dispatcher:
    _funs = {}
    key_stack = []

    @classmethod
    def key(cls):
        return cls.key_stack[-1] if cls.key_stack else None

    @classmethod
    def register_key(cls, key):
        if not key in cls._funs:
            cls._funs[key] = {}

    @classmethod
    def push_key(cls, key):
        if key not in cls._funs:
            raise RuntimeError(f"Key {key} not registered")
        cls.key_stack.append(key)

    @classmethod
    def pop_key(cls):
        cls.key_stack.pop()

    @classmethod
    def clear_key(cls):
        cls.key_stack = []


def dispatch(fun):
    fun_name = fun.__name__

    # @functools.wraps(fun)
    class _dispatcher:
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            if Dispatcher.key() is None:
                return fun(*args, **kwargs)
            else:
                return Dispatcher._funs[Dispatcher.key()][fun_name](*args, **kwargs)

        def register(self, key):
            Dispatcher.register_key(key)

            def _impl(fun):
                Dispatcher._funs[key][fun_name] = fun

            return _impl

    return _dispatcher()


# @dispatch
# def foo():
#     print("foo")


# @foo.register("tracing")
# def bar():
#     print("bar")


# @foo.register("other")
# def baz():
#     print("baz")


# # def foo():
# #     ...


# print(dispatch.__name__)
# print(foo.__name__)


# foo()
# Dispatcher.push_key("tracing")
# foo()
# Dispatcher.push_key("other")
# foo()
# Dispatcher.pop_key()
# foo()
# Dispatcher.clear_key()
# foo()
