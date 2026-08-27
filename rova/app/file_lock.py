from __future__ import annotations

import os
from pathlib import Path
import time


class FileLockError(RuntimeError):
    """Expected failure while acquiring or releasing a local file lock."""


class FileLock:
    """Small cross-platform advisory lock for one local storage root."""

    def __init__(self, path: Path, *, timeout_seconds: float = 10.0) -> None:
        self.path = Path(path)
        self.timeout_seconds = timeout_seconds
        self._handle = None

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.release()

    def acquire(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.path.open("a+b")
        except OSError as error:
            raise FileLockError("could not open file lock") from error
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                _try_lock(handle)
                self._handle = handle
                return
            except OSError:
                if time.monotonic() >= deadline:
                    handle.close()
                    raise FileLockError("timed out acquiring file lock")
                time.sleep(0.02)

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            _unlock(self._handle)
        except OSError as error:
            raise FileLockError("could not release file lock") from error
        finally:
            self._handle.close()
            self._handle = None


def _try_lock(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
