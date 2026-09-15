"""Безопасные операции с общей JSON-очередью WhatsApp на Windows.

Все писатели и потребитель очереди должны использовать только эти функции.
Файловый замок предотвращает потерю сообщения, когда два звонка завершаются
одновременно, а атомарная замена не даёт читателю увидеть половину JSON.
"""

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

try:
    import msvcrt
except ImportError:  # pragma: no cover - проект работает на Windows
    msvcrt = None


@contextmanager
def _queue_lock(queue_file: Path, timeout_seconds: float = 5.0):
    """Межпроцессный замок для конкретного файла очереди."""
    lock_file = queue_file.with_suffix(queue_file.suffix + ".lock")
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_file, "a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()

        if msvcrt is None:
            yield
            return

        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Не удалось заблокировать очередь: {queue_file}")
                time.sleep(0.05)

        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _load_queue(queue_file: Path) -> list:
    if not queue_file.exists():
        return []
    try:
        with open(queue_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _write_queue_atomically(queue_file: Path, items: list) -> None:
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = queue_file.with_name(
        f".{queue_file.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with open(temp_file, "w", encoding="utf-8") as handle:
            json.dump(items, handle, ensure_ascii=False, indent=4)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_file, queue_file)
    finally:
        try:
            temp_file.unlink(missing_ok=True)
        except OSError:
            pass


def append_queue_item(queue_file: Path, item: dict) -> int:
    """Добавляет одно сообщение и возвращает новый размер очереди."""
    queue_file = Path(queue_file)
    with _queue_lock(queue_file):
        queue = _load_queue(queue_file)
        queue.append(item)
        _write_queue_atomically(queue_file, queue)
        return len(queue)


def take_all_queue_items(queue_file: Path) -> list:
    """Атомарно забирает текущую пачку, оставляя новую очередь пустой.

    Сообщения, поступившие во время отправки этой пачки, будут добавлены в
    уже новую очередь и обработаются следующим циклом монитора.
    """
    queue_file = Path(queue_file)
    with _queue_lock(queue_file):
        queue = _load_queue(queue_file)
        if queue:
            _write_queue_atomically(queue_file, [])
        return queue
