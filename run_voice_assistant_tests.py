"""Автоматическая проверка записей и классификатора voice_assistant_v2.py."""
import ast
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
ISSUES_LOG = HERE / "voice_assistant_test_issues.md"
RESULTS_LOG = HERE / "voice_assistant_test_results.md"
TRANSFER_RESULTS_LOG = HERE / "voice_assistant_transfer_test_results.md"
TRANSFER_ISSUES_LOG = HERE / "voice_assistant_transfer_test_issues.md"
REQUESTS_DIR = HERE / "test_auto_request_mp3"
CONFIRMATIONS_DIR = HERE / "test_auto_confirmation_mp3"


def normalize(text):
    return re.sub(r"[^א-תa-z0-9]+", "", text.lower())


def load_cases():
    tree = ast.parse((HERE / "record_all_voices.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "AUTO_TEST_CASES" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("AUTO_TEST_CASES не найден")


def load_production_classifier():
    source_path = HERE / "voice_assistant_v2.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    classifier = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "ai_detect_intent"
    )
    namespace = {
        "json": json,
        "os": os,
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY"),
        "DEEPSEEK_URL": "https://api.deepseek.com/v1",
        "DEEPSEEK_MODEL": "deepseek-chat",
    }
    module = ast.Module(body=[classifier], type_ignores=[])
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["ai_detect_intent"]


def load_whisper():
    site_packages = r"C:\Users\david\AppData\Roaming\Python\Python312\site-packages"
    for path in (
        os.path.join(site_packages, "ctranslate2"),
        os.path.join(site_packages, "nvidia", "cublas", "bin"),
        os.path.join(site_packages, "nvidia", "cuda_runtime", "bin"),
    ):
        if os.path.exists(path):
            try:
                os.add_dll_directory(path)
            except OSError:
                pass
    from faster_whisper import WhisperModel
    print("📥 Загрузка Whisper для 50 аудиотестов...")
    return WhisperModel("ivrit-ai/whisper-large-v3-turbo-ct2", device="cuda", compute_type="int8_float16")


def transcribe(model, audio_path):
    segments, _ = model.transcribe(
        str(audio_path), language="he", beam_size=1, best_of=1,
        temperature=0.0, condition_on_previous_text=False,
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=300),
        word_timestamps=False,
    )
    return " ".join(segment.text for segment in segments).strip()


def matches_expected(result, expected):
    if expected == "separate":
        return result.get("intent") == "separate"
    if expected == "info":
        return result.get("intent") == "info"
    return result.get("intent") == expected


def main():
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("В .env отсутствует DEEPSEEK_API_KEY")
    cases = load_cases()
    transfer_only = "--transfer-only" in sys.argv
    classify = load_production_classifier()
    model = load_whisper()
    issues = []
    results = []

    if not transfer_only:
        for number, (name, expected_text, expected_intent) in enumerate(cases, start=1):
            audio_path = REQUESTS_DIR / f"{name}_kokoro.mp3"
            if not audio_path.exists():
                issues.append(f"{number}. `{name}` — нет аудиофайла `{audio_path.name}`.")
                continue
            transcribed = transcribe(model, audio_path)
            transcript_ok = normalize(transcribed) == normalize(expected_text)
            result = classify(transcribed) if transcribed else {"intent": "empty"}
            intent_ok = matches_expected(result, expected_intent)
            state = "PASS" if transcript_ok and intent_ok else "FAIL"
            results.append({
                "case": number, "file": audio_path.name, "expected_text": expected_text,
                "transcribed": transcribed, "expected_intent": expected_intent,
                "actual": result, "state": state,
            })
            print(f"[{number:02d}/50] {state}: {name} → {result}")
            if not transcript_ok:
                issues.append(
                    f"{number}. `{audio_path.name}` — Whisper: `{transcribed}`, ожидалось: `{expected_text}`."
                )
            if not intent_ok:
                issues.append(
                    f"{number}. `{audio_path.name}` — intent: `{result}`, ожидалось: `{expected_intent}`."
                )

    confirmation_expectations = {
        "yes_kokoro.mp3": "transfer",
        "correct_kokoro.mp3": "transfer",
        "no_kokoro.mp3": "end",
        "not_correct_kokoro.mp3": "end",
    }
    for filename, expected_answer in confirmation_expectations.items():
        audio_path = CONFIRMATIONS_DIR / filename
        transcribed = transcribe(model, audio_path) if audio_path.exists() else ""
        result = classify(transcribed, mode="transfer_response") if transcribed else {"answer": "empty"}
        ok = result.get("answer") == expected_answer
        print(f"[transfer] {filename}: {'PASS' if ok else 'FAIL'} → {result}")
        if not ok:
            issues.append(f"transfer `{filename}` — `{result}`, ожидалось `{expected_answer}`.")

    manual_cases = [
        ("Опечатка spa", "אני רוצה להזמיו עיסוי", "info"),
        ("Вопрос в transfer", "מתי ארוחת הבוקר?", "other"),
        ("Смешанный запрос", "אני רוצה מגבות ולשאול מתי יש ארוחת בוקר", "separate"),
    ]
    for label, text, expected in manual_cases:
        mode = "transfer_response" if expected == "other" else "main"
        result = classify(text, mode=mode)
        actual = result.get("answer") if mode != "main" else result.get("intent")
        ok = actual == expected
        print(f"[manual] {label}: {'PASS' if ok else 'FAIL'} → {result}")
        if not ok:
            issues.append(f"{label}: `{text}` → `{result}`, ожидалось `{expected}`.")

    required_files = [
        HERE / "Kokoro_RECORD" / "transfer_answer_required_kokoro.mp3",
        HERE / "ElevenLabs_RECORD" / "transfer_answer_required.mp3",
        HERE / "Kokoro_RECORD" / "separate_request_or_question_kokoro.mp3",
        HERE / "ElevenLabs_RECORD" / "separate_request_or_question.mp3",
    ]
    for path in required_files:
        if not path.exists() or path.stat().st_size == 0:
            issues.append(f"Отсутствует системная запись: `{path.name}`.")

    title = "Voice assistant transfer test results" if transfer_only else "Voice assistant test results"
    lines = [f"# {title}", "", f"Дата: {datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    lines.extend(
        f"- {row['state']}: `{row['file']}` → `{row['actual']}`" for row in results
    )
    results_path = TRANSFER_RESULTS_LOG if transfer_only else RESULTS_LOG
    issues_path = TRANSFER_ISSUES_LOG if transfer_only else ISSUES_LOG
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    issue_lines = [f"# {title.replace('results', 'issues')}", "", f"Дата: {datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    issue_lines.extend(f"- {issue}" for issue in issues) if issues else issue_lines.append("- Несоответствий не найдено.")
    issues_path.write_text("\n".join(issue_lines) + "\n", encoding="utf-8")
    print(f"\n📄 Результаты: {results_path.name}")
    print(f"📄 Несоответствия: {issues_path.name}")
    if not transfer_only:
        print(f"Итог: {len(results) - sum(row['state'] == 'FAIL' for row in results)}/{len(results)} основных аудиотестов прошли.")


if __name__ == "__main__":
    main()
