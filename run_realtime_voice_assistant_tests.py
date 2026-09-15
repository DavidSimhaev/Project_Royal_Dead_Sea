"""Сквозной realtime-тест: Kokoro MP3 -> микрофон -> voice_assistant_v2.py."""
import ast
import os
import queue
import random
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pygame

# PowerShell/Windows Console может открыться в cp1251. В live-выводе есть
# иврит и emoji, поэтому без UTF-8 поток чтения может погибнуть прямо во время теста.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
VOICE_SCRIPT = HERE / "voice_assistant_v2.py"
REQUESTS_DIR = HERE / "test_auto_request_mp3"
CONFIRMATIONS_DIR = HERE / "test_auto_confirmation_mp3"
LOG_PATH = HERE / "voice_assistant_realtime_test_log.md"
ISSUES_PATH = HERE / "voice_assistant_realtime_test_issues.md"
CONSOLE_PATH = HERE / "voice_assistant_realtime_console.log"
START_SIGNAL_PATH = HERE / ".realtime_tests_start.signal"
PYTHON = r"C:\Program Files\Python312\python.exe"


def random_test_room():
    """Возвращает существующий номер комнаты для тестовых WhatsApp-сообщений.

    Это используется только тестером. В рабочем звонке номер комнаты всегда
    вводится оператором и никогда не подменяется случайным значением.
    """
    floor = random.randint(6, 18)
    room_suffix = random.choice(
        [*range(1, 5), *range(20, 35), *range(50, 55), *range(70, 85)]
    )
    return str(floor * 100 + room_suffix)


def load_cases():
    tree = ast.parse((HERE / "record_all_voices.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "AUTO_TEST_CASES" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("AUTO_TEST_CASES не найден")


class Reporter:
    def __init__(self):
        self.results = []
        self.issues = []
        self.started = datetime.now()
        self.flush()

    def add(self, name, expected, actual, status, details=""):
        self.results.append((name, expected, actual, status, details))
        if status == "FAIL":
            self.issues.append(f"`{name}` — ожидалось `{expected}`, получено `{actual}`. {details}".strip())
        self.flush()

    def flush(self):
        lines = ["# Voice assistant realtime test log", "", f"Начат: {self.started:%Y-%m-%d %H:%M:%S}", ""]
        lines.extend(
            f"- {status}: `{name}` — expected `{expected}`, actual `{actual}`. {details}".rstrip()
            for name, expected, actual, status, details in self.results
        )
        LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        issue_lines = ["# Voice assistant realtime test issues", ""]
        issue_lines.extend(f"- {issue}" for issue in self.issues) if self.issues else issue_lines.append("- Пока несоответствий не найдено.")
        ISSUES_PATH.write_text("\n".join(issue_lines) + "\n", encoding="utf-8")


class VoiceSession:
    def __init__(self, room, wait_for_codex_start=False):
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self.process = subprocess.Popen(
            # -u нужен: иначе приглашение номера комнаты остаётся в буфере,
            # и realtime-тестер не может понять, что приложение готово.
            [PYTHON, "-u", str(VOICE_SCRIPT)], cwd=HERE,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=flags,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        self.lines = []
        self.events = queue.Queue()
        threading.Thread(target=self._read_output, daemon=True).start()
        # input("Номер комнаты: ") не добавляет перевод строки, поэтому ждём
        # предыдущий заголовок, который приложение печатает отдельной строкой.
        self.wait_for("ВВЕДИТЕ НОМЕР КОМНАТЫ", 45)
        self.process.stdin.write(f"{room}\n")
        self.process.stdin.flush()
        self.wait_for("🎤 Микрофон активен. Говорите...", 90)
        if wait_for_codex_start:
            wait_for_codex_start_signal()
        time.sleep(5)

    def _read_output(self):
        for line in self.process.stdout:
            clean = line.rstrip()
            self.lines.append(clean)
            self.events.put(clean)
            print(clean, flush=True)
            with CONSOLE_PATH.open("a", encoding="utf-8") as log_file:
                log_file.write(clean + "\n")

    def wait_for(self, phrase, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if any(phrase in line for line in self.lines):
                return True
            if self.process.poll() is not None:
                output = "\n".join(self.lines[-20:]) or "консоль не успела вывести текст"
                raise RuntimeError(f"Ассистент завершился до '{phrase}':\n{output}")
            time.sleep(0.1)
        raise TimeoutError(f"Не дождались '{phrase}'")

    def play(self, path):
        start = len(self.lines)
        sound = pygame.mixer.Sound(str(path))
        pygame.mixer.Channel(0).play(sound)
        while pygame.mixer.Channel(0).get_busy():
            time.sleep(0.1)
        return start

    def wait_for_after(self, phrase, start, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if any(phrase in line for line in self.lines[start:]):
                return True
            if self.process.poll() is not None:
                return False
            time.sleep(0.15)
        return False

    def wait_result(self, start, timeout=45, transfer=False):
        deadline = time.time() + timeout
        pattern = "📋 AI ответ:" if transfer else "📋 Намерение:"
        while time.time() < deadline:
            for line in self.lines[start:]:
                if not transfer and "📶 Интернет:" in line:
                    return line
                if pattern in line:
                    return line
            if self.process.poll() is not None:
                return "PROCESS_EXIT"
            time.sleep(0.15)
        return "TIMEOUT"

    def wait_for_response_completion(self, start, timeout=45):
        """Проверяет не только intent, но и факт успешного завершения озвучивания."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            output = "\n".join(self.lines[start:])
            if "⚠️ ELEVENLABS:" in output:
                return "ELEVENLABS_QUOTA_STOP"
            # Сам заголовок маршрута появляется до сохранения записи и отправки
            # в WhatsApp. Ждём финальную строку перед штатным завершением child.
            if (
                "доделать логику когда программа будет готова" in output
                and ("🚨 URGENT:" in output or "⚠️ COMPLAINT:" in output)
            ):
                return "TRANSFERRED"
            if "🎤 Микрофон возобновлён" in output:
                return "SPOKEN"
            if "✅ Запись воспроизведена:" in output:
                return "RECORDED"
            if self.process.poll() is not None:
                return "PROCESS_EXIT"
            time.sleep(0.15)
        return "TIMEOUT"

    def stop(self):
        if self.process.poll() is not None:
            return
        try:
            # После возврата микрофона ответ уже полностью прозвучал. Обычное
            # завершение не печатает ложный forrtl/control-BREAK error в консоль.
            self.process.terminate()
            self.process.wait(timeout=12)
        except Exception:
            self.process.kill()
            self.process.wait(timeout=8)


def wait_for_codex_start_signal():
    """Pause only before the first test recording until Codex receives the user's signal."""
    print("⏸️ Первый микрофон активен. Ожидаю сигнал из Codex перед первой записью...", flush=True)
    while not START_SIGNAL_PATH.exists():
        time.sleep(0.2)
    START_SIGNAL_PATH.unlink(missing_ok=True)
    print("▶️ Сигнал получен. Запускаю записи.", flush=True)


def expected_in_console(expected, line, name=""):
    if name == "17_wifi_problem":
        return (
            "📶 Интернет: wifi.mp3 (eleven)" in line
            or "📶 Интернет: wifi_kokoro.mp3 (kokoro)" in line
        )
    if expected == "info":
        return "Намерение: info" in line
    return f"Намерение: {expected}" in line


def test_regular_cases(cases, reporter, wait_for_codex_start=False):
    """Каждый файл идёт в отдельный живой звонок без истории прошлых гостей."""
    for index, (name, _text, expected) in enumerate(cases):
        number = int(name[:2])
        session = VoiceSession(
            random_test_room(),
            wait_for_codex_start=wait_for_codex_start and index == 0,
        )
        try:
            start = session.play(REQUESTS_DIR / f"{name}_kokoro.mp3")
            line = session.wait_result(start)

            completion = session.wait_for_response_completion(start, 45)
            output = "\n".join(session.lines[start:])
            not_understood = "לא הצלחתי להבין" in output
            status = "PASS" if (
                expected_in_console(expected, line, name)
                and not not_understood
                and (
                    completion in {"SPOKEN", "RECORDED"}
                    or (expected in {"urgent", "complaint"} and completion == "TRANSFERRED")
                )
            ) else "FAIL"
            details = ""
            if not_understood:
                details = "Распознано намерение, но обработчик ответил «не удалось понять»."
            elif completion == "ELEVENLABS_QUOTA_STOP":
                details = "Намерение распознано, но ElevenLabs не озвучил ответ: сработал стоп-сценарий токенов."
            elif completion not in {"SPOKEN", "RECORDED", "TRANSFERRED"}:
                details = f"Озвучивание не завершилось: {completion}."
            reporter.add(f"{number:02d}_{name}", expected, line, status, details)
            # Не закрываем звонок посреди озвучивания ответа. Это также
            # подтверждает, что микрофон вернулся к приёму после полного AI/TTS-ответа.
            if completion == "SPOKEN":
                session.wait_for_after("Микрофон возобновлён", start, 45)
        finally:
            session.stop()


def test_regular_cases_batch(cases, reporter):
    """Быстрый live-прогон маршрутизации без перезагрузки Whisper для каждого MP3."""
    session = VoiceSession(random_test_room())
    try:
        for name, _text, expected in cases:
            number = int(name[:2])
            start = session.play(REQUESTS_DIR / f"{name}_kokoro.mp3")
            line = session.wait_result(start)
            status = "PASS" if expected_in_console(expected, line, name) else "FAIL"
            reporter.add(f"{number:02d}_{name}", expected, line, status)

            # Не перекрываем ответ ассистента следующей записью.  Для request
            # ждём реальный возврат микрофона; информационные записи статичны.
            if "Намерение: request" in line:
                if not session.wait_for_after("Микрофон возобновлён", start, 45):
                    time.sleep(4)
            else:
                time.sleep(7)
    finally:
        session.stop()


def test_transfer_case(number, name, expected_answer, reporter):
    session = VoiceSession(random_test_room())
    try:
        start = session.play(REQUESTS_DIR / f"{name}_kokoro.mp3")
        initial = session.wait_result(start)
        reporter.add(f"{number:02d}_{name}_start", "transfer", initial,
                     "PASS" if "Намерение: transfer" in initial else "FAIL")
        session.wait_for_after("Микрофон возобновлён", start, 45)

        start = session.play(REQUESTS_DIR / "32_breakfast_kokoro.mp3")
        other = session.wait_result(start, transfer=True)
        required = any("Transfer: нужен ответ да или нет" in line for line in session.lines[start:])
        reporter.add(f"{number:02d}_{name}_other_question", "other + transfer_answer_required", other,
                     "PASS" if "AI ответ: other" in other and required else "FAIL")
        session.wait_for_after("Микрофон возобновлён", start, 45)

        filename = "correct_kokoro.mp3" if expected_answer == "transfer" else "not_correct_kokoro.mp3"
        start = session.play(CONFIRMATIONS_DIR / filename)
        answer = session.wait_result(start, transfer=True)
        reporter.add(f"{number:02d}_{name}_confirmation", expected_answer, answer,
                     "PASS" if f"AI ответ: {expected_answer}" in answer else "FAIL")
        session.wait_for_after("Микрофон возобновлён", start, 45)
    finally:
        session.stop()


def main():
    pygame.mixer.init()
    START_SIGNAL_PATH.unlink(missing_ok=True)
    CONSOLE_PATH.write_text(
        f"Realtime console started: {datetime.now():%Y-%m-%d %H:%M:%S}\n",
        encoding="utf-8",
    )
    reporter = Reporter()
    cases = load_cases()
    regular = [case for case in cases if case[2] != "transfer"]
    info_transfer_only = "--info-transfer" in sys.argv
    wait_for_codex_start = "--wait-for-codex-start" in sys.argv
    wifi_only = "--wifi-only" in sys.argv
    wifi_then_remaining = "--wifi-then-remaining" in sys.argv
    resume_from = next((int(arg.split("=", 1)[1]) for arg in sys.argv if arg.startswith("--from=")), None)
    only_numbers = next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--only=")), None)
    if wifi_only:
        regular = [case for case in regular if case[0] == "17_wifi_problem"]
    if wifi_then_remaining:
        remaining_numbers = set(range(21, 31)) | set(range(37, 50)) | {17}
        regular = [
            case for index, case in enumerate(cases, start=1)
            if index in remaining_numbers and case[2] != "transfer"
        ]
    if info_transfer_only:
        regular = [case for case in regular if case[2] == "info"]
    if resume_from is not None:
        regular = [case for case in regular if int(case[0].split("_", 1)[0]) >= resume_from]
    if only_numbers:
        selected_numbers = {int(number) for number in only_numbers.split(",") if number.strip()}
        regular = [case for case in regular if int(case[0].split("_", 1)[0]) in selected_numbers]
    limit = 20 if "--limit-20" in sys.argv else None
    if limit:
        regular = regular[:limit]
    if "--batch" in sys.argv:
        test_regular_cases_batch(regular, reporter)
    else:
        test_regular_cases(regular, reporter, wait_for_codex_start=wait_for_codex_start)
    if limit:
        print(f"Готово: {len(reporter.results)} realtime-проверок.")
        print(f"Логи: {LOG_PATH.name}, {ISSUES_PATH.name}")
        return
    if wifi_only or wifi_then_remaining or only_numbers:
        print(f"Готово: {len(reporter.results)} realtime-проверок.")
        print(f"Логи: {LOG_PATH.name}, {ISSUES_PATH.name}")
        return
    transfer_cases = [case for case in cases if case[2] == "transfer"]
    for index, (name, _text, _expected) in enumerate(transfer_cases, start=1):
        confirmation = "transfer" if index % 2 else "end"
        number = next(i for i, case in enumerate(cases, start=1) if case[0] == name)
        test_transfer_case(number, name, confirmation, reporter)
    print(f"Готово: {len(reporter.results)} realtime-проверок.")
    print(f"Логи: {LOG_PATH.name}, {ISSUES_PATH.name}")


if __name__ == "__main__":
    main()
