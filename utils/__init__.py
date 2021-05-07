from functools import partial, reduce


def none_func(value):
    return None


def inspect_wrapper(fn):
    def inner(value):
        fn(value)
        return value

    return inner


def get_emotion_code_from_description(emotions):
    def inner(descriptions):
        result_array = []
        for key, value in emotions.items():
            if value in descriptions:
                result_array.append(key)
        return result_array

    return inner


def curr_reduce(fn):
    def inner(initializer):
        def inner_deep(iterable):
            return reduce(fn, iterable, initializer)

        return inner_deep

    return inner
