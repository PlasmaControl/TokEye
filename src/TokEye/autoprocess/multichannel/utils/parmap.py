"""Defines function parmap, a parallel version of map."""

import gc
import multiprocessing as mp
import sys
from collections.abc import Sequence
from ctypes import c_ulong
from multiprocessing.sharedctypes import Value
from typing import Any, Callable


def _identity(x: Any) -> Any:
    """The identity function."""
    return x


class ProgressBar:
    """
    Usage : update_progress(progress)
    An ASCII progression bar.
    Values of input should be floats or ints ranging from 0 to ntot
    (default 1).
    """

    def __init__(
        self,
        output: Callable[[str], None] | None = None,
        bar_length: int = 50,
        ntot: float | None = None,
    ) -> None:
        """
        "output" is a function that takes a string as input and prints it
        (defaults to stdout).
        bar_length is the length of the progress bar in characters.
        ntot is the maximum value of the progress bar (default 1).
        """
        self.output = output
        self.bar_length = bar_length
        self.ntot = ntot
        self._last = -1

    def set_ntot(self, value: float) -> None:
        self.ntot = value

    def _update(self, progress: float | int) -> None:
        status = ""
        try:
            progress = float(progress)
        except (ValueError, TypeError):
            progress = 0
            status = "error: progress var must be float\r\n"
        if self.ntot is not None:
            progress /= self.ntot
        if progress < self._last:
            status = "Non monotonic"
        self._last = progress
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(self.bar_length * progress))
        text = "\rPercent: [{0}] {1:.3f}% {2}"
        text = text.format(
            "#" * block + "-" * (self.bar_length - block),
            progress * 100,
            status,
        )
        if self.output is None:
            sys.stdout.write(text)
            sys.stdout.flush()
        else:
            self.output(text)

    def __call__(self, progress: float | int) -> None:
        """Update the progress bar to specified value."""
        self._update(progress)


class ParallelMapper:
    """Parallel version of map."""

    def __init__(
        self,
        nprocs: int | None = None,
        progress: bool | ProgressBar = True,
        fetcher: Callable[[Any], Any] = _identity,
    ) -> None:
        """
        nprocs is the number of processors to use (default is mp.cpu_count()).
        progress is a boolean or a ProgressBar object. If True, a ProgressBar
        is used.
        fetcher is a function that fetches the data from the iterable, defaults
        to identity function.
        """
        if nprocs is None:
            nprocs = mp.cpu_count()
        self.nprocs = nprocs
        self.fetcher = fetcher
        self._counter = Value(c_ulong, 0)
        self._set_counter(0)
        if progress is True:
            progress = ProgressBar()
        self._progress = progress

    def progress(self, value: float | int, ntot: float | None = None) -> None:
        """Update the progress bar to specified value."""
        if self._progress:
            if ntot and value == 0:
                self._progress.set_ntot(ntot)
            self._progress(value)

    def _set_counter(self, value: int) -> None:
        """Set the counter to value."""
        with self._counter.get_lock():
            self._counter.value = value

    def _get_counter(self) -> int:
        """Return the current value of the counter."""
        with self._counter.get_lock():
            out = self._counter.value
        return out

    def _inc_counter(self, value: int = 1) -> int:
        """Increment the counter by value and return the new value."""
        with self._counter.get_lock():
            self._counter.value += value
            out = self._counter.value
        return out

    def _fun(
        self,
        f: Callable[..., Any],
        q_in: mp.Queue,
        q_out: mp.Queue,
        **kwargs: Any,
    ) -> None:
        """Function to be executed by each process.
        Wrapper for f and progress bar.
        """
        while True:
            i, x = q_in.get()
            if i is None:
                break
            q_out.put((i, f(self.fetcher(x), **kwargs)))
            # Try to free memory
            gc.collect()
            self.progress(self._inc_counter())

    def parmap(
        self,
        f: Callable[..., Any],
        x: Sequence[Any],
        **kwargs: Any,
    ) -> list[Any]:
        """
        Parallelizes the computation of 'map(f, X)' over 'nprocs' processors,
        where f is a function, X is an iterable. Returns list of length len(X).
        """
        self.progress(0, len(x))
        if self.nprocs == 1:
            return list(map(f, map(self.fetcher, x)))

        # Create input/output queues
        q_in = mp.Queue(1)
        q_out = mp.Queue()

        # Create and start processes
        proc = [
            mp.Process(target=self._fun, args=(f, q_in, q_out), kwargs=kwargs)
            for _ in range(self.nprocs)
        ]
        for p in proc:
            p.daemon = True
            p.start()

        # Feed/execute processes
        sent = [q_in.put((i, x)) for i, x in enumerate(x)]
        [q_in.put((None, None)) for _ in range(self.nprocs)]
        res = [q_out.get() for _ in range(len(sent))]

        [p.join() for p in proc]

        # compile and return results
        out = [x for i, x in sorted(res)]
        for p in proc:
            p.close()
            del p
        del (proc, res, sent, q_in, q_out)
        gc.collect()
        return out

    def __call__(
        self, f: Callable[..., Any], x: Sequence[Any], **kwargs: Any
    ) -> list[Any]:
        """
        Parallelizes the computation of 'map(f, X)' over 'nprocs' processors,
        where f is a function, X is an iterable. Returns list of length len(X).
        """
        return self.parmap(f, x, **kwargs)


def parmap(
    f: Callable[..., Any],
    x: Sequence[Any],
    nprocs: int | None = None,
    **kwargs: Any,
) -> list[Any]:
    """
    Parallelizes the computation of 'map(f, x)' over 'nprocs' processors,
    where f is a function, x is an iterable. Returns list of length len(x).
    """
    mapper = ParallelMapper(nprocs=nprocs, **kwargs)
    return mapper(f, x)
