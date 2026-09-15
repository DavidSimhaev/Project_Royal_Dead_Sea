"""Локальная проверка двух одновременных звонков без настоящего WhatsApp."""

import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "Project"


def simulate_call(queue_path_str: str, room: str, request: str) -> int:
    """Запускается в отдельном процессе, как независимый телефонный звонок."""
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from queue_utils import append_queue_item

    return append_queue_item(Path(queue_path_str), {
        "id": f"simulation-{room}",
        "room": room,
        "text": f"SIMULATION: room {room} requested {request}",
        "kind": "simulation",
    })


def main():
    with tempfile.TemporaryDirectory(prefix="voice_assistant_two_calls_") as temp_dir:
        queue_path = Path(temp_dir) / "send_queue.json"
        calls = [("1302", "מגבות"), ("672", "שמפו")]
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(simulate_call, str(queue_path), room, request) for room, request in calls]
            for future in futures:
                future.result(timeout=10)

        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        received = {(item["room"], item["text"]) for item in queue}
        expected = {
            ("1302", "SIMULATION: room 1302 requested מגבות"),
            ("672", "SIMULATION: room 672 requested שמפו"),
        }
        assert received == expected, (received, expected)
        print("PASS: 2 simultaneous simulated calls kept separate")
        for item in sorted(queue, key=lambda value: value["room"]):
            print(f"  room={item['room']}: {item['text']}")


if __name__ == "__main__":
    main()
