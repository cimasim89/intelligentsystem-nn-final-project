def none_func(value):
    return None


def print_wrapper(fn):
    def inner(value):
        fn(value)
        return value

    return inner
