"""Проверки лимитов ElevenLabs без запуска микрофона, API и аудиоустройств."""
import ast
import types
from datetime import datetime
from pathlib import Path


SOURCE = Path(__file__).with_name("voice_assistant_v2.py")
FUNCTIONS = {
    "get_today_str", "get_hotel_elevenlabs_status", "set_hotel_kokoro_for_today",
    "set_elevenlabs_tokens_depleted", "clear_elevenlabs_tokens_depleted",
    "get_room_engine", "update_room_chars", "set_kokoro_cooldown",
    "end_call_after_elevenlabs_limit", "elevenlabs_quota_error",
    "exceeds_daily_limit", "speak_elevenlabs", "speak_kokoro",
}


class ExitCall(BaseException):
    pass


class FakeOS:
    def _exit(self, code):
        raise ExitCall(code)


class FakeTime:
    def time(self):
        return 1_700_000_000

    def sleep(self, _seconds):
        pass


def load_functions(storage, events):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in FUNCTIONS]
    ns = {
        "datetime": datetime,
        "Path": Path,
        "time": FakeTime(),
        "os": FakeOS(),
        "MAX_CHARS_PER_CALL": 250,
        "MAX_CHARS_PER_DAY": 5000,
        "KOKORO_COOLDOWN_HOURS": 6,
        "HOTEL_STATUS_KEY": "__hotel_elevenlabs__",
        "ROOM_NUMBER": "101",
        "VOICE_ENGINE": "eleven",
        "JESSICA_VOICE_ID": "test_voice",
        "ELEVENLABS_API_KEY": "test_key",
        "ELEVENLABS_MODEL": "eleven_v3",
        "chars_this_call": 0,
        "STARTUP_ELEVENLABS_STOP_REASON": None,
        "load_room_status": lambda: storage,
        "save_room_status": lambda _status: events.append("status:save"),
        "elevenlabs_has_tokens": lambda: True,
        "get_audio_path": lambda key, engine=None: events.append(f"audio:{key}:{engine}") or "stop.mp3",
        "play_audio": lambda path: events.append(f"play:{path}"),
        "stop_ding_loop": lambda: events.append("ding:stop"),
        "stop_recording_and_save": lambda: events.append("recording:save"),
        "stop_music": lambda: events.append("music:stop"),
        "print": lambda *parts, **_kwargs: events.append("print:" + " ".join(map(str, parts))),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    ns["get_today_str"] = lambda: "2026-09-15"
    return ns


def expect_limit(ns, events, text, expected_flag):
    def stop(reason, **kwargs):
        events.append(f"limit:{reason}:{kwargs}")
        raise ExitCall()

    ns["end_call_after_elevenlabs_limit"] = stop
    try:
        ns["speak_elevenlabs"](text)
    except ExitCall:
        pass
    assert any(expected_flag in event for event in events), events


def main():
    # 250 за звонок: завершить текущий звонок и дать комнате Kokoro на 6 часов.
    storage, events = {}, []
    ns = load_functions(storage, events)
    ns["chars_this_call"] = 245
    expect_limit(ns, events, "123456", "'room_cooldown': True")

    # 5000 за день: считать по отелю, завершить звонок и включить Kokoro всем.
    storage = {"__hotel_elevenlabs__": {"chars_today": 4995, "last_reset_day": "2026-09-15"}}
    events = []
    ns = load_functions(storage, events)
    expect_limit(ns, events, "123456", "'hotel_daily_limit': True")

    # Отельный лимит действует на новую комнату до смены дня.
    storage = {"__hotel_elevenlabs__": {"chars_today": 5000, "last_reset_day": "2026-09-15"}}
    events = []
    ns = load_functions(storage, events)
    assert ns["get_room_engine"]("202") == "kokoro"

    # Нет токенов при старте: первая попытка планирует stop+exit, далее Kokoro.
    storage, events = {}, []
    ns = load_functions(storage, events)
    ns["elevenlabs_has_tokens"] = lambda: False
    assert ns["get_room_engine"]("101") == "kokoro"
    assert ns["STARTUP_ELEVENLABS_STOP_REASON"] == "В ELEVENLABS НЕТ ТОКЕНОВ"
    ns["STARTUP_ELEVENLABS_STOP_REASON"] = None
    assert ns["get_room_engine"]("102") == "kokoro"
    assert ns["STARTUP_ELEVENLABS_STOP_REASON"] is None

    # ElevenLabs вернул 429: это не обычная ошибка, а завершение звонка через
    # ту же stop-запись с постоянным переключением на Kokoro.
    storage, events = {}, []
    ns = load_functions(storage, events)
    ns["requests"] = types.SimpleNamespace(
        post=lambda *_args, **_kwargs: types.SimpleNamespace(status_code=429, text="quota exceeded")
    )
    expect_limit(ns, events, "תגובה", "'tokens_depleted': True")

    # Kokoro не содержит счётчиков ElevenLabs и не вызывает обновление статуса.
    kokoro_node = next(node for node in ast.parse(SOURCE.read_text(encoding="utf-8")).body if getattr(node, "name", None) == "speak_kokoro")
    kokoro_source = ast.unparse(kokoro_node)
    assert "update_room_chars" not in kokoro_source
    assert "chars_this_call" not in kokoro_source

    # Общая ветка лимита проигрывает stop MP3, печатает маркер и завершает звонок.
    storage, events = {}, []
    ns = load_functions(storage, events)
    try:
        ns["end_call_after_elevenlabs_limit"]("TEST", room_cooldown=True, hotel_daily_limit=True, tokens_depleted=True)
    except ExitCall:
        pass
    assert "play:stop.mp3" in events
    assert any("доделать логику когда программа будет готова" in event for event in events)
    assert storage["__hotel_elevenlabs__"]["kokoro_for_day"] == "2026-09-15"
    assert storage["__hotel_elevenlabs__"]["tokens_depleted"] is True

    print("PASS: 250/call, 5000/hotel-day, no-tokens startup, Kokoro unlimited")


if __name__ == "__main__":
    main()
