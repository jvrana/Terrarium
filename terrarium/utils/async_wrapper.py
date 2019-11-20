import asyncio
from functools import partial
from functools import wraps

import nest_asyncio
import uvloop
from more_itertools import divide
from more_itertools import flatten
from tqdm import tqdm

# setup

nest_asyncio.apply()
uvloop.install()


def with_index(fxn):
    @wraps(fxn)
    def wrapped(i, *args, **kwargs):
        return (i, fxn(*args, **kwargs))

    return wrapped


async def exec_async_fxn(
    fxn, arg_list, kwargs, chunk_size=1, desc="", progress_bar=True
):
    """Executes an asynchronous function.

    :param fxn: function to execute in parallel
    :type fxn: callable
    :param arg_list: list of arguments as tuples for each function run
    :type arg_list: list of tuples
    :param kwargs: key-value arguments for each function run
    :type kwargs: dict
    :param chunk_size: number of functions to run for each worker
    :type chunk_size: int
    :param desc: description
    :type desc: basestring
    :param progress_bar: whether to display tqdm progress bar
    :type progress_bar: boolean
    :return: list of results in same order as arg_list
    :rtype: list
    """
    # run asynchronously

    loop = asyncio.get_event_loop()
    partial_fxn = partial(with_index(fxn), **kwargs)
    futures = [
        loop.run_in_executor(None, partial_fxn, i, *args)
        for i, args in enumerate(arg_list)
    ]

    # collect results
    results = []

    iterator = asyncio.as_completed(futures)
    if progress_bar:
        iterator = tqdm(iterator, desc=desc, total=len(futures), unit_scale=chunk_size)

    for f in iterator:
        results.append(await f)
    results = sorted(results, key=lambda x: x[0])
    return [r[1] for r in results]


def asyncfunc(fxn, arg_list, kwargs=None, chunk_size=1, progress_bar=True, desc=None):
    """Runs a function asynchronously.

    :param fxn: function to run asynchronously
    :type fxn: function or lambda
    :param arg_chunks: arguments to apply to the function; suggested to divide list
        into chunks
    :type arg_chunks: list
    :return: result
    :rtype: list
    """
    if kwargs is None:
        kwargs = {}
    # finish loop
    loop = asyncio.get_event_loop()
    if desc is None:
        desc = desc
    results = loop.run_until_complete(
        exec_async_fxn(
            fxn,
            arg_list,
            kwargs=kwargs,
            desc=desc,
            progress_bar=progress_bar,
            chunk_size=chunk_size,
        )
    )
    #     loop.close()
    return results


def make_async(
    chunk_size, progress_bar=True, as_classmethod=False, data_pos=0, return_type=list
):
    """Wrapper to make a function run asynchrounously.

    :param chunk_size: size of array to apply to each worker
    :type chunk_size: int
    :param progress_bar: whether to display a progress bar
    :type progress_bar: bool
    :param as_classmethod: whether to pass in the first argument as a instance for
        instance or classmethods
    :type as_classmethod: bool
    :param data_pos: position in arguments where list of data is
    :type data_pos: int
    :return: results
    :rtype: list
    """

    def dec(fxn):
        data_position = data_pos
        if as_classmethod:
            data_position += 1

        @wraps(fxn)
        def wrapper(*args, **kwargs):
            data = args[data_position]
            post_data_args = args[data_position + 1 :]
            pre_data_args = args[:data_position]
            chunks = divide(chunk_size, data)
            arg_list = [pre_data_args + (c,) + post_data_args for c in chunks]
            desc = 'Running "{}" [size: {}, num: {}]: '.format(
                fxn.__name__, chunk_size, len(chunks)
            )
            results = asyncfunc(
                fxn, arg_list, kwargs=kwargs, progress_bar=progress_bar, desc=desc
            )
            if all([r is None for r in results]):
                return None
            try:
                iter_results = flatten(results)
            except TypeError:
                iter_results = results
            if return_type:
                return return_type(iter_results)
            return results

        return wrapper

    return dec
