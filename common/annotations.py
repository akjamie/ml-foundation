from typing import Callable
import os


def proxy(value: str, http_proxy="", https_proxy=""):
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # both http & https share the same proxy url
            if value is not None:
                os.environ["https_proxy"] = value
                os.environ["http_proxy"] = value
                os.environ["no_proxy"] = "localhost,127.0.0.1"

            # todo: to support http_proxy & https_proxy

            return func(*args, **kwargs)

        return wrapper

    return decorator
