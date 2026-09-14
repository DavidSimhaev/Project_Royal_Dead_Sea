# voice_assistant_v2.py
import os
import sys
import time
import tempfile
import requests
import json
import asyncio
import threading
import queue
import pyaudio
import wave
import re
import urllib.request
from pathlib import Path
from datetime import datetime
from collections import deque
import random
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

import pygame
import numpy as np
import speech_recognition as sr
import edge_tts

# ============================================================
# GROQ (быстрый Whisper)
# ============================================================
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    print("❌ Groq не установлен! pip install groq")
    GROQ_AVAILABLE = False

# Fallback на IVRIT
try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# ============================================================
# PYGAME
# ============================================================
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
music_channel = pygame.mixer.Channel(0)
ding_channel = pygame.mixer.Channel(1)
voice_channel = pygame.mixer.Channel(2)
loading_channel = pygame.mixer.Channel(3)
voice_channel.set_volume(1.0)

# ============================================================
# ПУТИ
# ============================================================
HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
os.chdir(HERE)
sys.path.insert(0, str(HERE))
LOADING_SOUND = HERE / "phone-beeps.mp3"
ROOM_STATUS_FILE = HERE / "room_status.json"

# ============================================================
# НАСТРОЙКИ
# ============================================================
MAX_CHARS_PER_CALL = 250
MAX_CHARS_PER_DAY = 5000
KOKORO_COOLDOWN_HOURS = 6
SILENCE_INTERVAL = 30
PRE_BUFFER_SIZE = 10
ENERGY_THRESHOLD = 700
SILENCE_THRESHOLD = 12
MIN_TEXT_LENGTH = 2
SPEAK_PAUSE_BEFORE = 0.15
SPEAK_PAUSE_AFTER = 0.5
ECHO_PROTECTION_TIME = 0.8

# ⭐ GROQ API KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = None

# Fallback IVRIT
WHISPER_MODEL_NAME = "ivrit-ai/whisper-large-v3-turbo"
WHISPER_DEVICE = "cpu"
whisper_pipe = None
whisper_model = None

YES_WORDS = ["כן", "נכון", "בסדר", "אוקיי", "אוקי", "בטח", "יאללה"]
NO_WORDS = ["לא", "לא תודה", "לא צריך", "לא, תודה", "לא רוצה"]

# ============================================================
# ЗАГРУЗКА WHISPER
# ============================================================
def load_whisper_model():
    global groq_client, whisper_pipe, whisper_model
    # ⭐ Сначала пробуем Groq
    if GROQ_AVAILABLE:
        try:
            groq_client = Groq(api_key=GROQ_API_KEY)
            print(f"✅ Groq клиент готов (whisper-large-v3)")
            return
        except Exception as e:
            print(f"⚠️ Groq ошибка: {e}")
            groq_client = None
    # ⭐ Fallback на IVRIT
    if WHISPER_AVAILABLE:
        print(f"📥 Загрузка IVRIT Whisper (fallback)...")
        try:
            from transformers import pipeline
            whisper_pipe = pipeline(
                "automatic-speech-recognition",
                model=WHISPER_MODEL_NAME,
                device=WHISPER_DEVICE,
                chunk_length_s=30,
            )
            print(f"✅ IVRIT Whisper загружен")
        except Exception as e:
            print(f"❌ IVRIT не загрузился: {e}")
            try:
                import whisper as _w
                whisper_model = _w.load_model("medium", device=WHISPER_DEVICE)
                print(f"✅ Fallback Whisper medium загружен")
            except Exception as e2:
                print(f"❌ Всё упало: {e2}")

# ============================================================
# ГЛОБАЛЬНЫЕ
# ============================================================
audio_queue = queue.Queue()
is_speaking = False
microphone_active = False
mic_thread = None
stop_mic = False
mic_paused = False
conversation_history = []
MAX_HISTORY = 20
stop_loading_ding = False
ding_thread_running = False
recording_frames = []
is_recording = False
recording_lock = threading.Lock()
current_conversation_text = []
transfer_to_staff_mode = False
spa_transfer_mode = False
room_service_mode = False
staff_mode_active = False
chars_this_call = 0
verdict_played = False
all_requests = []
last_activity_time = time.time()
silence_thread = None
stop_silence_monitor = False

# ============================================================
# ДИНЬ-ЦИКЛ
# ============================================================
def start_ding_loop():
    global stop_loading_ding, ding_thread_running
    if ding_thread_running:
        return
    if not Path("ding.mp3").exists():
        return
    stop_loading_ding = False
    ding_thread_running = True

    def _loop():
        global ding_thread_running
        count = 0
        try:
            sound = pygame.mixer.Sound("ding.mp3")
            sound.set_volume(0.35)
            ding_channel.play(sound)
            count += 1
            print(f"🔔 ДИНЬ #{count}")
        except Exception as e:
            print(f"⚠️ ding ошибка: {e}")
            ding_thread_running = False
            return
        while not stop_loading_ding:
            for _ in range(20):
                if stop_loading_ding:
                    print(f"🔕 ДИНЬ СТОП (всего: {count})")
                    ding_thread_running = False
                    return
                time.sleep(0.1)
            try:
                sound = pygame.mixer.Sound("ding.mp3")
                sound.set_volume(0.35)
                ding_channel.play(sound)
                count += 1
                print(f"🔔 ДИНЬ #{count}")
            except Exception:
                break
        print(f"🔕 ДИНЬ СТОП (всего: {count})")
        ding_thread_running = False

    threading.Thread(target=_loop, daemon=True).start()


def stop_ding_loop():
    global stop_loading_ding
    stop_loading_ding = True

# ============================================================
# play_audio
# ============================================================
def play_audio(path, volume=1.0):
    stop_ding_loop()
    try:
        if not Path(path).exists():
            print(f"❌ Не найден: {path}")
            return False
        sound = pygame.mixer.Sound(str(path))
        sound.set_volume(volume)
        voice_channel.play(sound)
        while voice_channel.get_busy():
            time.sleep(0.05)
        voice_channel.stop()
        return True
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

# ============================================================
# ФИЛЬТР
# ============================================================
HALLUCINATIONS = {
    "אה", "אהה", "הא", "הממ", "אההה", "אה, אה", "אה אה",
    "רוסית", "עברית", "אנגלית",
    "כתוביות", "כתובית", "כתוביות בעברית",
    "תרגום", "עריכה", "סוף", "סוף.",
    "מוזיקה", "מוסיקה", "מוזיקה.",
    "אממ", "אממ.",
    "...", "…", ".", "..",
}

def filter_whisper_hallucinations(text):
    if not text:
        return ""
    clean = text.strip().rstrip('.!?,;:')
    if len(clean) < 2:
        print(f"🚫 Фильтр: коротко '{text}'")
        return ""
    if clean in HALLUCINATIONS:
        print(f"🚫 Фильтр: галлюцинация '{text}'")
        return ""
    words = clean.split()
    if len(words) >= 3:
        unique_words = set(words)
        if len(unique_words) == 1:
            print(f"🚫 Фильтр: повтор '{text}'")
            return ""
        if len(unique_words) <= 2 and len(words) >= 5:
            print(f"🚫 Фильтр: монотонно '{text}'")
            return ""
    if re.match(r'^[אה\s,.\-]+$', clean):
        print(f"🚫 Фильтр: 'אה' '{text}'")
        return ""
    if re.match(r'^(.)\1{3,}$', clean):
        print(f"🚫 Фильтр: повтор символа '{text}'")
        return ""
    if not re.search(r'[א-תa-zA-Z]', clean):
        print(f"🚫 Фильтр: нет букв '{text}'")
        return ""
    return text

# ============================================================
# ЗВУК ЗАГРУЗКИ
# ============================================================
def start_loading_sound():
    try:
        if LOADING_SOUND.exists():
            sound = pygame.mixer.Sound(str(LOADING_SOUND))
            sound.set_volume(0.4)
            loading_channel.play(sound, loops=-1)
            print(f"🔔 Звук загрузки: {LOADING_SOUND.name}")
            return True
        return False
    except Exception as e:
        print(f"⚠️ Ошибка: {e}")
        return False

def stop_loading_sound():
    try:
        loading_channel.stop()
        print("🔕 Звук загрузки остановлен")
    except:
        pass

# ============================================================
# ROOM_STATUS
# ============================================================
def load_room_status():
    if ROOM_STATUS_FILE.exists():
        try:
            with open(ROOM_STATUS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Ошибка: {e}")
    return {}

def save_room_status(status):
    try:
        with open(ROOM_STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"⚠️ Ошибка: {e}")

def get_today_str():
    return datetime.now().strftime("%Y-%m-%d")

def get_room_engine(room_number):
    status = load_room_status()
    room = status.get(str(room_number), {})
    now = time.time()
    kokoro_until = room.get("kokoro_until", 0)
    if kokoro_until > now:
        remaining_min = int((kokoro_until - now) / 60)
        hours = remaining_min // 60
        minutes = remaining_min % 60
        print(f"🎤 Комната {room_number}: KOKORO (осталось {hours}ч {minutes}мин)")
        return 'kokoro'
    today = get_today_str()
    if room.get("last_reset_day") != today:
        room["chars_today"] = 0
        room["last_reset_day"] = today
        room["chars_this_call"] = 0
        status[str(room_number)] = room
        save_room_status(status)
        print(f"🔄 Комната {room_number}: сброс")
    chars_today = room.get("chars_today", 0)
    if chars_today >= MAX_CHARS_PER_DAY:
        print(f"🎤 Комната {room_number}: KOKORO (лимит)")
        return 'kokoro'
    print(f"🎤 Комната {room_number}: ELEVENLABS ({chars_today}/{MAX_CHARS_PER_DAY})")
    return 'eleven'

def update_room_chars(room_number, chars, is_call_end=False):
    status = load_room_status()
    room = status.get(str(room_number), {})
    today = get_today_str()
    if room.get("last_reset_day") != today:
        room["chars_today"] = 0
        room["last_reset_day"] = today
    room["chars_this_call"] = room.get("chars_this_call", 0) + chars
    room["chars_today"] = room.get("chars_today", 0) + chars
    if is_call_end:
        room["chars_this_call"] = 0
    status[str(room_number)] = room
    save_room_status(status)
    return room["chars_this_call"], room["chars_today"]

def set_kokoro_cooldown(room_number):
    status = load_room_status()
    room = status.get(str(room_number), {})
    kokoro_until = time.time() + (KOKORO_COOLDOWN_HOURS * 3600)
    room["kokoro_until"] = kokoro_until
    room["chars_this_call"] = 0
    status[str(room_number)] = room
    save_room_status(status)
    end_time = datetime.fromtimestamp(kokoro_until).strftime('%H:%M')
    print(f"🧊 Комната {room_number}: Kokoro {KOKORO_COOLDOWN_HOURS}ч (до {end_time})")

# ============================================================
# ВВОД КОМНАТЫ
# ============================================================
def ask_room_number():
    print("=" * 60)
    print("🏨 ВВЕДИТЕ НОМЕР КОМНАТЫ")
    print("=" * 60)
    room_number = input("Номер комнаты: ").strip()
    if not room_number:
        room_number = "test_default"
        print(f"⚠️ Не введён, использую: {room_number}")
    engine = get_room_engine(room_number)
    return room_number, engine

ROOM_NUMBER, VOICE_ENGINE = ask_room_number()

print("\n⏳ Загрузка модели...")
loading_started = start_loading_sound()
load_whisper_model()

if VOICE_ENGINE == 'kokoro':
    AUDIO_DIR = HERE / "Kokoro_RECORD"
    VOICE_SUFFIX = "_kokoro"
    print("🎤 KOKORO")
else:
    AUDIO_DIR = HERE / "ElevenLabs_RECORD"
    VOICE_SUFFIX = ""
    print("🎤 ELEVENLABS")

# ============================================================
# KOKORO
# ============================================================
KOKORO_BASE = Path("C:/Users/david/Desktop/kokoro-hebrew-main/inference")
KOKORO_MODEL_PATH = KOKORO_BASE / "kokoro_v1_hebrew.pth"
KOKORO_VOICE_PATH = KOKORO_BASE / "voices" / "he_shaul.pt"

km = None
voice = None
phonemize_hebrew = None
g2p = None

if VOICE_ENGINE == 'kokoro':
    print("📥 Загрузка Kokoro...")
    try:
        import soundfile as sf
        from kokoro import KModel
        device = "cpu"
        km = KModel(repo_id="hexgrad/Kokoro-82M", model=str(KOKORO_MODEL_PATH)).to(device).eval()
        voice = torch.load(KOKORO_VOICE_PATH, map_location=device, weights_only=True)
        sys.path.insert(0, str(KOKORO_BASE))
        try:
            from hebrew_g2p import phonemize_hebrew as ph
            phonemize_hebrew = ph
        except:
            def phonemize_hebrew(text):
                return text, ""
        try:
            from renikud_onnx import G2P
            renikud_model_path = HERE / "renikud_model.onnx"
            if not renikud_model_path.exists():
                urllib.request.urlretrieve(
                    "https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx",
                    renikud_model_path
                )
            g2p = G2P(str(renikud_model_path))
            print("✅ ReNikud загружен")
        except Exception as e:
            print(f"⚠️ ReNikud: {e}")
            g2p = None
        print("✅ Kokoro загружен")
    except Exception as e:
        print(f"❌ Ошибка Kokoro: {e}")
        stop_loading_sound()
        sys.exit(1)

if loading_started:
    stop_loading_sound()

# ============================================================
# API KEYS
# ============================================================
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
JESSICA_VOICE_ID = "cgSgspJ2msm6clMCkdW9"
ELEVENLABS_MODEL = "eleven_flash_v2_5"   # ⭐ быстрая модель
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# ============================================================
# ОБЩИЕ
# ============================================================
MUSIC_VOLUME = 0.05
GREETING_DELAY = 1.5
CONVERSATIONS_DIR = "conversations"
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

# ============================================================
# АУДИО БАЗА
# ============================================================
AUDIO_DATABASE_BASE = {
    "transfer_confirm": "transfer_confirm",
    "transfer_complete": "transfer_complete",
    "not_understood": "not_understood",
    "transfer_complete_SPA": "transfer_complete_SPA",
    "continue_conversation": "continue_convershion",
    "info_breakfast": "info_breakfast",
    "info_lunch": "info_lunch",
    "info_dinner": "info_dinner",
    "info_all_meals": "info_all_meals",
    "info_lobby": "info_lobby",
    "info_pool": "info_pool",
    "info_beach": "info_beach",
    "info_spa": "info_spa",
    "info_spa_transfer": "info_spa_transfer",
    "info_synagogue": "info_synagogue",
    "info_rules": "info_rules",
    "info_room_service": "room_service",
    "info_wifi": "wifi",
    "first_remember": "first_remember",
    "second_remember": "second_remember",
    "last_remember": "last_remember",
    "confirm_end_request": "confirm_end_request",
    "end_message_request": "end_message_request",
}

def get_audio_path(intent_key, engine=None):
    if engine is None:
        engine = VOICE_ENGINE
    if intent_key == "greeting":
        path = (HERE / "Kokoro_RECORD" / "greeting_kokoro.mp3") if engine == 'kokoro' else (HERE / "ElevenLabs_RECORD" / "greeting_elevenlabs.mp3")
        return path if path.exists() else None
    if intent_key == "error":
        path = (HERE / "Kokoro_RECORD" / "error_kokoro.mp3") if engine == 'kokoro' else (HERE / "ElevenLabs_RECORD" / "error_elevenlabs.mp3")
        return path if path.exists() else None
    if intent_key == "stop_request_and_changing_line":
        path = HERE / "ElevenLabs_RECORD" / "stop_request_and_changing_line.mp3"
        return path if path.exists() else None
    base_name = AUDIO_DATABASE_BASE.get(intent_key)
    if not base_name:
        return None
    suffix = "_kokoro" if engine == 'kokoro' else ""
    audio_dir = HERE / "Kokoro_RECORD" if engine == 'kokoro' else HERE / "ElevenLabs_RECORD"
    path = audio_dir / f"{base_name}{suffix}.mp3"
    if path.exists():
        return path
    print(f"⚠️ Файл не найден: {path}")
    return None

# ============================================================
# WHATSAPP
# ============================================================
QUEUE_FILE = Path(r"C:\Users\david\Desktop\WhatsApp\Project\send_queue.json")

def send_verdict_to_whatsapp(requests_text, audio_file_path=None):
    print("🔍 send_verdict_to_whatsapp!")
    if not requests_text:
        requests_text = "בקשה לא זוהתה"
    message = f"""🎤 *הודעה מהמערכת הדיגיטלית* (AI)

📋 *האורח מ-{ROOM_NUMBER} ביקש:*
{requests_text}"""
    queue_data = []
    if QUEUE_FILE.exists():
        try:
            with open(QUEUE_FILE, "r", encoding="utf-8") as f:
                queue_data = json.load(f)
        except:
            queue_data = []
    queue_data.append({
        "id": str(int(time.time() * 1000)),
        "text": message,
        "audio_file": audio_file_path,
        "created_at": time.time()
    })
    try:
        with open(QUEUE_FILE, "w", encoding="utf-8") as f:
            json.dump(queue_data, f, ensure_ascii=False, indent=4)
        print(f"📤 В очередь (всего: {len(queue_data)})")
        return True
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

# ============================================================
# МОНИТОР МОЛЧАНИЯ
# ============================================================
def silence_monitor():
    global last_activity_time, stop_silence_monitor
    print("🔕 Монитор молчания (60/90)")
    warning_2_played = False
    warning_3_played = False
    check_start = time.time()
    last_known_activity = last_activity_time
    while not stop_silence_monitor:
        try:
            if is_speaking or voice_channel.get_busy():
                check_start = time.time()
                last_known_activity = last_activity_time
                time.sleep(0.5)
                continue
            if last_activity_time > last_known_activity:
                print(f"🔄 Сброс таймера")
                last_known_activity = last_activity_time
                check_start = time.time()
                warning_2_played = False
                warning_3_played = False
                continue
            elapsed = time.time() - check_start
            if elapsed >= SILENCE_INTERVAL * 2:
                if not warning_2_played:
                    print(f"🔔 Молчание 60с → second_remember")
                    warning_2_played = True
                    path = get_audio_path("second_remember")
                    if path:
                        play_audio(path)
                    check_start = time.time()
                    continue
            if elapsed >= SILENCE_INTERVAL * 3:
                if not warning_3_played:
                    print(f"🔔 Молчание 90с → ЗАВЕРШЕНИЕ")
                    warning_3_played = True
                    path = get_audio_path("last_remember")
                    if path:
                        play_audio(path)
                    try:
                        stop_recording_and_save()
                    except:
                        pass
                    update_room_chars(ROOM_NUMBER, 0, is_call_end=True)
                    print("👋 Завершение по молчанию...")
                    stop_music()
                    stop_ding_loop()
                    time.sleep(1)
                    os._exit(0)
            time.sleep(0.5)
        except Exception as e:
            print(f"⚠️ Ошибка: {e}")
            time.sleep(1)

def start_silence_monitor():
    global silence_thread, stop_silence_monitor, last_activity_time
    stop_silence_monitor = False
    last_activity_time = time.time()
    silence_thread = threading.Thread(target=silence_monitor, daemon=True)
    silence_thread.start()

def update_activity():
    global last_activity_time
    last_activity_time = time.time()

# ============================================================
# ЗАПИСЬ РАЗГОВОРА
# ============================================================
def stop_recording_and_save():
    global recording_frames, current_conversation_text, is_recording
    print("⏹️ Сохранение записи...")
    frames_copy = []
    text_copy = []
    try:
        with recording_lock:
            frames_copy = recording_frames.copy()
            text_copy = current_conversation_text.copy()
            recording_frames = []
            current_conversation_text = []
    except Exception as e:
        print(f"⚠️ Ошибка: {e}")
        return None
    if not frames_copy:
        print("⚠️ Нет данных")
        return None
    print(f"📊 Сохранение {len(frames_copy)} фреймов...")
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"conversation_{timestamp}"
        mp3_path = os.path.abspath(os.path.join(CONVERSATIONS_DIR, f"{base_filename}.mp3"))
        txt_path = os.path.abspath(os.path.join(CONVERSATIONS_DIR, f"{base_filename}.txt"))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b''.join(frames_copy))
        try:
            import subprocess
            subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path, "-acodec", "libmp3lame", "-ab", "64k", mp3_path],
                capture_output=True, check=True, timeout=5
            )
            print(f"✅ Запись: {mp3_path}")
            os.remove(wav_path)
        except Exception as e:
            print(f"⚠️ Ошибка конвертации: {e}")
            wav_final = os.path.join(CONVERSATIONS_DIR, f"{base_filename}.wav")
            os.rename(wav_path, wav_final)
            mp3_path = wav_final
        if text_copy:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Диалог {timestamp}\n")
                f.write(f"Комната: {ROOM_NUMBER}\n")
                f.write("=" * 50 + "\n")
                for line in text_copy:
                    f.write(line + "\n")
            print(f"✅ Текст: {txt_path}")
        return mp3_path
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None

def add_conversation_text(speaker, text):
    with recording_lock:
        timestamp = datetime.now().strftime("%H:%M:%S")
        current_conversation_text.append(f"[{timestamp}] {speaker}: {text}")

# ============================================================
# МУЗЫКА
# ============================================================
def start_music():
    try:
        if not Path("background.mp3").exists():
            print("⚠️ background.mp3 не найден")
            return
        music_channel.play(pygame.mixer.Sound("background.mp3"), loops=-1)
        music_channel.set_volume(MUSIC_VOLUME)
        print(f"🎵 Музыка ({int(MUSIC_VOLUME * 100)}%)")
    except Exception as e:
        print(f"⚠️ Ошибка музыки: {e}")

def stop_music():
    try:
        music_channel.stop()
    except:
        pass

# ============================================================
# ИСПРАВЛЕНИЕ ОШИБОК
# ============================================================
def fix_recognition_errors(text):
    corrections = {
        "הרוחות": "השעות",
        "רוחות": "שעות",
        "הרוח": "השעה",
        "הרוחת": "השעות",
        "מוגבות": "מגבות",
        "ממקבות": "מגבות",
        "לחדור": "לחדר",
        "להגדר": "לחדר",
    }
    original_text = text
    for wrong, right in corrections.items():
        if wrong in text:
            text = text.replace(wrong, right)
            print(f"🔧 '{wrong}' → '{right}'")
    if text != original_text:
        print(f"✏️ После: {text}")
    return text

# ============================================================
# ПРОВЕРКИ
# ============================================================
def is_audio_busy():
    return (
        mic_paused
        or is_speaking
        or voice_channel.get_busy()
    )

def is_yes(text):
    clean = text.strip().rstrip('.!?,;:').lower()
    for word in YES_WORDS:
        if clean == word or clean.startswith(word + " ") or clean.startswith(word + ","):
            return True
    return False

def is_no(text):
    clean = text.strip().rstrip('.!?,;:').lower()
    for word in NO_WORDS:
        if clean == word or clean.startswith(word + " ") or clean.startswith(word + ","):
            return True
    return False

# ============================================================
# МИКРОФОН
# ============================================================
def mic_worker():
    global microphone_active, stop_mic, mic_paused, is_recording, recording_frames
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    p = pyaudio.PyAudio()
    stream = None
    last_speech_end = 0
    pause_release_time = time.time()
    pre_buffer = deque(maxlen=PRE_BUFFER_SIZE)
    print("🎤 mic_worker ЗАПУЩЕН")
    with recording_lock:
        is_recording = True
    while not stop_mic:
        try:
            if stream is None:
                stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
                microphone_active = True
                pre_buffer.clear()
                print("🎤 Микрофон открыт")
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_array = np.frombuffer(data, dtype=np.int16)
            if len(audio_array) == 0:
                continue
            if not mic_paused and not is_speaking:
                with recording_lock:
                    if is_recording:
                        recording_frames.append(data)
            mean_square = np.mean(audio_array.astype(np.float32) ** 2)
            if mean_square <= 0:
                continue
            energy = np.sqrt(mean_square)

            if staff_mode_active:
                pre_buffer.clear()
                time.sleep(0.1)
                continue

            if is_audio_busy():
                pre_buffer.clear()
                pause_release_time = time.time()
                time.sleep(0.05)
                continue
            if time.time() - pause_release_time < ECHO_PROTECTION_TIME:
                pre_buffer.append(data)
                time.sleep(0.05)
                continue
            pre_buffer.append(data)
            if energy > ENERGY_THRESHOLD and (time.time() - last_speech_end) > 0.8:
                frames = list(pre_buffer)
                silence_count = 0
                aborted = False
                for _ in range(80):
                    if is_audio_busy():
                        print("⏸️ Пауза — прерываю")
                        aborted = True
                        break
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    if len(audio_array) == 0:
                        continue
                    mean_square = np.mean(audio_array.astype(np.float32) ** 2)
                    energy = np.sqrt(mean_square) if mean_square > 0 else 0
                    if energy > ENERGY_THRESHOLD:
                        frames.append(data)
                        silence_count = 0
                    else:
                        silence_count += 1
                        frames.append(data)
                        if silence_count > SILENCE_THRESHOLD:
                            break
                last_speech_end = time.time()
                if aborted:
                    pre_buffer.clear()
                    continue
                if is_audio_busy():
                    print("⏸️ Пауза перед Whisper")
                    pre_buffer.clear()
                    continue
                if len(frames) < 16:
                    print(f"⚠️ Короткая запись ({len(frames)})")
                    pre_buffer.clear()
                    continue

                start_ding_loop()
                print("🔔 ДИНЬ запущен")

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                    wav_path = tmp_wav.name
                    with wave.open(wav_path, 'wb') as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(p.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        wf.writeframes(b''.join(frames))

                text = ""
                # ⭐ GROQ — БЫСТРО
                if groq_client is not None:
                    try:
                        print(f"🎤 Groq Whisper (быстро)...")
                        with open(wav_path, "rb") as audio_file:
                            transcription = groq_client.audio.transcriptions.create(
                                file=audio_file,
                                model="whisper-large-v3",
                                language="he",
                                response_format="text",
                                temperature=0.0,
                            )
                        if isinstance(transcription, str):
                            text = transcription.strip()
                        else:
                            text = transcription.text.strip()
                        print(f"🔍 Groq: '{text}'")
                        text = filter_whisper_hallucinations(text)
                        if not text:
                            print("🚫 Отфильтрован")
                    except Exception as e:
                        print(f"⚠️ Groq ошибка: {e}")
                        text = ""

                # Fallback IVRIT
                if not text and whisper_pipe is not None:
                    try:
                        print(f"🎤 IVRIT Whisper (fallback)...")
                        result = whisper_pipe(
                            wav_path,
                            generate_kwargs={"language": "he", "task": "transcribe"},
                            return_timestamps=False,
                        )
                        text = result["text"].strip()
                        print(f"🔍 Whisper: '{text}'")
                        text = filter_whisper_hallucinations(text)
                    except Exception as e:
                        print(f"⚠️ Whisper: {e}")
                        text = ""

                if not text:
                    try:
                        recognizer = sr.Recognizer()
                        with sr.AudioFile(wav_path) as source:
                            audio = recognizer.record(source)
                            text = recognizer.recognize_google(audio, language="he-IL")
                        print(f"🔍 Google: '{text}'")
                        text = filter_whisper_hallucinations(text)
                    except:
                        text = ""

                try:
                    os.remove(wav_path)
                except:
                    pass

                if text:
                    text = fix_recognition_errors(text)
                    if len(text) >= MIN_TEXT_LENGTH:
                        print(f"🎤 ✅ Распознано: {text}")
                        audio_queue.put(text)
                        add_conversation_text("Гость", text)
                        update_activity()
                    else:
                        stop_ding_loop()
                        print("🚫 Слишком коротко — динь стоп")
                else:
                    stop_ding_loop()
                    print("🚫 Пусто — динь стоп")

                pre_buffer.clear()
        except Exception as e:
            print(f"⚠️ Ошибка mic_worker: {e}")
            time.sleep(0.1)
    with recording_lock:
        is_recording = False
    if stream:
        stream.stop_stream()
        stream.close()
    p.terminate()
    microphone_active = False
    print("🎤 mic_worker завершён")

def start_microphone():
    global mic_thread, stop_mic
    if mic_thread and mic_thread.is_alive():
        print("⚠️ mic_worker уже работает")
        return
    stop_mic = False
    mic_thread = threading.Thread(target=mic_worker, daemon=True)
    mic_thread.start()


def stop_microphone():
    global mic_paused
    mic_paused = True
    print("⏸️ Микрофон на паузе")


def resume_microphone():
    global mic_paused
    mic_paused = False
    print("▶️ Микрофон возобновлён")

# ============================================================
# KOKORO SPEAK
# ============================================================
def speak_kokoro(text):
    try:
        if g2p:
            try:
                text_with_niqqud = g2p.phonemize(text)
            except:
                text_with_niqqud = text
        else:
            text_with_niqqud = text
        SPLIT = re.compile(r"(?<=[.!?])\s+")
        pieces = []
        for chunk in SPLIT.split(text_with_niqqud.strip()):
            if not chunk:
                continue
            ps, dropped = phonemize_hebrew(chunk)
            if not ps:
                continue
            n = min(len(ps), voice.shape[0]) - 1
            with torch.no_grad():
                out = km(ps, voice[n], speed=1.0, return_output=True)
            pieces.append(out.audio.cpu().numpy())
        if pieces:
            import soundfile as sf
            audio_data = np.concatenate(pieces)
            audio_data = audio_data * 5.0
            audio_data = np.clip(audio_data, -1.0, 1.0)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = tmp.name
                sf.write(output_path, audio_data, 24000)
            sound = pygame.mixer.Sound(output_path)
            voice_channel.play(sound)
            while voice_channel.get_busy():
                time.sleep(0.05)
            voice_channel.stop()
            try:
                os.remove(output_path)
            except:
                pass
            return True
        else:
            return speak_fallback(text)
    except Exception as e:
        print(f"❌ Ошибка Kokoro: {e}")
        return speak_fallback(text)

# ============================================================
# ELEVENLABS SPEAK
# ============================================================
def speak_elevenlabs(text, language="he"):
    global chars_this_call
    if not text or len(text.strip()) < 2:
        print(f"⚠️ Пустой текст")
        return False, 0
    try:
        clean_text = text.replace('"', '').replace("'", "").replace("\n", " ").strip()[:300]
        chars_to_use = len(clean_text)
        if chars_this_call + chars_to_use > MAX_CHARS_PER_CALL:
            print(f"⚠️ Превышение: {chars_this_call} + {chars_to_use} > {MAX_CHARS_PER_CALL}")
            return False, 0
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{JESSICA_VOICE_ID}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept-Charset": "utf-8"
        }
        data = {
            "text": clean_text,
            "model_id": ELEVENLABS_MODEL,
            "voice_settings": {
                "stability": 0.4,
                "similarity_boost": 0.8,
                "style": 0.5,
                "use_speaker_boost": True
            }
        }
        if language == "he":
            data["language_code"] = "he"
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(response.content)
                temp_path = tmp.name
            sound = pygame.mixer.Sound(temp_path)
            voice_channel.play(sound)
            while voice_channel.get_busy():
                time.sleep(0.05)
            voice_channel.stop()
            try:
                os.remove(temp_path)
            except:
                pass
            chars_this_call += chars_to_use
            update_room_chars(ROOM_NUMBER, chars_to_use)
            remaining = MAX_CHARS_PER_CALL - chars_this_call
            print(f"💰 ElevenLabs: +{chars_to_use} ({chars_this_call}/{MAX_CHARS_PER_CALL}, осталось: {remaining})")
            return True, chars_to_use
        else:
            print(f"❌ ElevenLabs: {response.status_code}")
            return speak_fallback(clean_text, language), 0
    except Exception as e:
        print(f"❌ ElevenLabs: {e}")
        return speak_fallback(text, language), 0


def speak_fallback(text, language="he"):
    try:
        if not text or len(text.strip()) < 2:
            return False
        communicate = edge_tts.Communicate(text, "he-IL-HilaNeural", rate="+5%", volume="+10%")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            temp_path = tmp.name
            asyncio.run(communicate.save(temp_path))
        sound = pygame.mixer.Sound(temp_path)
        voice_channel.play(sound)
        while voice_channel.get_busy():
            time.sleep(0.05)
        voice_channel.stop()
        try:
            os.remove(temp_path)
        except:
            pass
        return True
    except Exception as e:
        print(f"❌ Fallback: {e}")
        return False

# ============================================================
# ЛИМИТ
# ============================================================
def handle_limit_exceeded():
    print("=" * 60)
    print(f"⚠️ ЛИМИТ {MAX_CHARS_PER_CALL} СИМВОЛОВ!")
    print("=" * 60)
    stop_ding_loop()
    stop_path = get_audio_path("stop_request_and_changing_line")
    if stop_path:
        play_audio(stop_path)
    set_kokoro_cooldown(ROOM_NUMBER)
    update_room_chars(ROOM_NUMBER, 0, is_call_end=True)
    try:
        stop_recording_and_save()
    except:
        pass
    print("👋 Завершение...")
    stop_music()
    stop_ding_loop()
    time.sleep(1)
    os._exit(0)

# ============================================================
# ФИНАЛЬНЫЙ ВЕРДИКТ
# ============================================================
def play_final_verdict(requests_text):
    global chars_this_call
    print("=" * 60)
    print(f"✅ ФИНАЛЬНЫЙ ВЕРДИКТ")
    print(f"📋 Просьбы: {requests_text}")
    print("=" * 60)
    stop_ding_loop()
    intro_path = get_audio_path("confirm_end_request")
    if intro_path:
        play_audio(intro_path)
    if VOICE_ENGINE == 'eleven':
        chars_needed = len(requests_text)
        if chars_this_call + chars_needed > MAX_CHARS_PER_CALL:
            handle_limit_exceeded()
        else:
            speak_elevenlabs(requests_text)
    else:
        speak_kokoro(requests_text)
    outro_path = get_audio_path("end_message_request")
    if outro_path:
        time.sleep(0.8)
        play_audio(outro_path)
    audio_path = None
    try:
        audio_path = stop_recording_and_save()
    except:
        pass
    send_verdict_to_whatsapp(requests_text, audio_path)
    update_room_chars(ROOM_NUMBER, 0, is_call_end=True)
    print("👋 Завершение...")
    stop_music()
    stop_ding_loop()
    time.sleep(1)
    os._exit(0)

# ============================================================
# SPEAK
# ============================================================
def speak(text, language="he"):
    global is_speaking, chars_this_call, verdict_played, mic_paused
    if not text or len(text.strip()) < 2:
        print(f"⚠️ Пустой текст")
        is_speaking = False
        return
    print(f"🤖 AI [{VOICE_ENGINE}]: {text}")
    stop_ding_loop()
    mic_paused = True
    is_speaking = True
    time.sleep(SPEAK_PAUSE_BEFORE)
    add_conversation_text("ИИ", text)
    is_verdict = "שיהיה לכם חופשה מהנה" in text
    if is_verdict:
        pattern = r'העברתי את בקשתכם לצוות:\s*(.+?)\s*\.\s*שיהיה לכם חופשה מהנה'
        match = re.search(pattern, text)
        requests_text = match.group(1).strip() if match else text
        is_speaking = False
        verdict_played = True
        play_final_verdict(requests_text)
        return
    if VOICE_ENGINE == 'kokoro':
        speak_kokoro(text)
    else:
        chars_needed = len(text)
        if chars_this_call + chars_needed > MAX_CHARS_PER_CALL:
            handle_limit_exceeded()
        else:
            speak_elevenlabs(text, language)
    time.sleep(SPEAK_PAUSE_AFTER)
    is_speaking = False
    mic_paused = False
    print("🎤 Микрофон возобновлён")
    update_activity()

# ============================================================
# AI С ПАМЯТЬЮ
# ============================================================
def ask_ai_with_memory(prompt, language="he"):
    global conversation_history
    print("=" * 60)
    print("🔍 ДИАГНОСТИКА AI")
    print("=" * 60)
    print(f"📝 prompt: '{prompt}'")
    print(f"📚 История: {len(conversation_history)}")
    print(f"📋 Уже собрано просьб: {all_requests}")

    current_requests_text = ", ".join(all_requests) if all_requests else "אין"

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_URL,
            timeout=15.0
        )

        system_prompt = f"""אתה העוזר הדיגיטלי של מלון רויאל ים המלח.

⚠️ תמיד תחזיר תשובת JSON בלבד!

{{
  "understood": true,
  "reply": "הטקסט שתרצה לומר לאורח",
  "requests": ["מגבות", "שמפו"]
}}

או אם לא הבנת:
{{
  "understood": false,
  "reply": "סלחו לי, אשמח להבין בדיוק מה אתם צריכים.",
  "requests": []
}}

📌 בקשות שכבר נאספו בשיחה: {current_requests_text}

📌 הכללים:

1. אתה זוכר מה האורח ביקש קודם — все בקשות уже в списке выше.
2. Если гость добавляет НОВУЮ просьбу — добавь её в "requests".
3. Если гость повторяет — не дублируй.
4. Если гость говорит "כן"/"לא"/"תודה" — "requests": [].
5. Если не понял — understood: false.
6. После 2 непонятых — "מצטער. לא הצלחתי להבין."

⚠️⚠️⚠️ שדה "requests" — רשימת בקשות **נקיות** בעברית תקנית:
- "אפשר מגבות לחדר" → ["מגבות לחדר"]
- "אני צריך מרכך שמפו ומגבות" → ["מרכך", "שמפו", "מגבות"]
- "לא תודה" → []
- "כן" → []
- "מתי ארוחת בוקר" → []

⚠️⚠️⚠️ פסק דין סופי (только когда гость сказал כן/לא תודה):
"העברתי את בקשתכם לצוות: [כל הבקשות]. שיהיה לכם חופשה מהנה!"
requests: []

📝 תבניות:

🔹 Первый ответ:
"בשמחה נשלח לכם [רשימת בקשות]. האם אתם רוצים עוד משהו?"

🔹 Второй ответ:
"הבנתי. אז אתם רוצים [בקשות 1] ו-[בקשות 2], נכון?"

🔹 Финал:
"העברתי את בקשתכם לצוות: [כל הבקשות]. שיהיה לכם חופשה מהנה!"

✅ חובה:
- לדבר בלשון רבים (לכם / אתם)
- תשובה максимум 60-70 תווים
- "requests" — תמיד רשימה (массив)
- вернуть JSON
"""

        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history[-10:]:
            messages.append(msg)
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=120,
            response_format={"type": "json_object"}
        )

        finish_reason = response.choices[0].finish_reason
        usage = response.usage
        raw_content = response.choices[0].message.content
        print(f"✅ finish_reason: {finish_reason}")
        print(f"📊 tokens: {usage.total_tokens}")

        if not raw_content:
            print(f"❌ Пустой content!")
            return "לא הצלחתי להבין, נסו שוב", False, []

        raw_answer = raw_content.strip()
        requests_list = []
        try:
            parsed = json.loads(raw_answer)
            understood = parsed.get("understood", True)
            reply = parsed.get("reply", raw_answer)
            requests_list = parsed.get("requests", [])
            if not isinstance(requests_list, list):
                requests_list = []
            requests_list = [r.strip() for r in requests_list if isinstance(r, str) and r.strip()]
            if not reply or len(reply.strip()) < 2:
                reply = "לא הצלחתי להבין, נסו שוב"
        except json.JSONDecodeError:
            understood = True
            reply = raw_answer if len(raw_answer) > 3 else "לא הצלחתי להבין, נסו שוב"
            requests_list = []

        conversation_history.append({"role": "user", "content": prompt})
        conversation_history.append({"role": "assistant", "content": reply})
        if len(conversation_history) > MAX_HISTORY:
            conversation_history = conversation_history[-MAX_HISTORY:]

        not_understood = not understood
        print(f"💬 reply: '{reply}'")
        print(f"📋 requests от AI: {requests_list}")
        print("=" * 60)

        return reply, not_understood, requests_list

    except Exception as e:
        print(f"❌ Ошибка AI: {e}")
        return "לא הצלחתי להבין, נסו שוב", False, []

# ============================================================
# AI-КЛАССИФИКАТОР
# ============================================================
def ai_detect_intent(user_text, mode="main"):
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_URL,
            timeout=10.0
        )
        if mode == "main":
            system_prompt = """אתה מסווג בקשות של אורחי מלון בעברית. תן תשובת JSON בלבד.

1. "transfer" - האורח רוצה לדבר עם נציג אנושי
2. "info" - שאלה על מידע:
   - breakfast, lunch, dinner, all_meals - ארוחות
   - lobby, pool, beach, spa, spa_transfer, synagogue, rules, wifi, room_service
3. "request" - כל דבר אחר

⚠️ "כן", "לא", "בסדר", "תודה" - תמיד "request"!

החזר JSON:
{"intent": "transfer"} / {"intent": "info", "topic": "wifi"} / {"intent": "request"}
"""
            user_message = user_text
        elif mode == "transfer_response":
            system_prompt = """הקשר: שאלו "האם אתם רוצים שאעביר אתכם לנציג אנושי?"

החזר תשובת JSON:
{"answer": "transfer"} - הסכמה
{"answer": "end"} - סירוב
{"answer": "other"} - אחר
"""
            user_message = f"תשובת האורח: {user_text}"
        elif mode == "spa_response":
            system_prompt = """הקשר: שאלו "האם אתם מעוניינים לדבר עם נציג צוות הספא?"
החזר JSON: {"answer": "transfer"} / {"answer": "end"} / {"answer": "other"}
"""
            user_message = f"תשובת האורח: {user_text}"
        elif mode == "room_service_response":
            system_prompt = """הקשר: שאלו "האם תרצו שאעביר אתכם לנציג אנושי?"
החזר JSON: {"answer": "transfer"} / {"answer": "end"} / {"answer": "other"}
"""
            user_message = f"תשובת האורח: {user_text}"
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=100,
            response_format={"type": "json_object"}
        )
        raw_content = response.choices[0].message.content
        if not raw_content or len(raw_content) < 5:
            if mode == "main":
                return {"intent": "request"}
            else:
                return {"answer": "other"}
        result = json.loads(raw_content)
        return result
    except Exception as e:
        print(f"⚠️ Ошибка классификатора: {e}")
        if mode == "main":
            return {"intent": "request"}
        else:
            return {"answer": "other"}

# ============================================================
# ЗАВЕРШЕНИЯ
# ============================================================
def play_transfer_complete():
    print("=" * 60)
    print("👤 ЗАВЕРШИТЬ ЛОГИКУ ПЕРЕВОДА НА ПЕРСОНАЛ")
    print("=" * 60)
    try:
        complete_path = get_audio_path("transfer_complete")
        if complete_path:
            play_audio(complete_path)
        try:
            stop_recording_and_save()
        except:
            pass
        update_room_chars(ROOM_NUMBER, 0, is_call_end=True)
        print("👋 Завершение...")
        stop_music()
        stop_ding_loop()
        time.sleep(1)
        os._exit(0)
    except Exception as e:
        print(f"❌ {e}")
        os._exit(1)

def play_spa_transfer_complete():
    print("=" * 60)
    print("🧖 ЗАВЕРШЕНИЕ ПЕРЕВОДА В СПА")
    print("=" * 60)
    try:
        complete_path = get_audio_path("transfer_complete_SPA")
        if complete_path:
            play_audio(complete_path)
        try:
            stop_recording_and_save()
        except:
            pass
        update_room_chars(ROOM_NUMBER, 0, is_call_end=True)
        print("👋 Завершение...")
        stop_music()
        stop_ding_loop()
        time.sleep(1)
        os._exit(0)
    except Exception as e:
        print(f"❌ {e}")
        os._exit(1)

def play_greeting_with_delay():
    global mic_paused
    try:
        greeting_path = get_audio_path("greeting")
        if not greeting_path:
            print(f"❌ Приветствие не найдено")
            return False
        print("🔵 Пауза микрофона")
        mic_paused = True
        time.sleep(0.5)
        print(f"🔵 Ожидание {GREETING_DELAY}с")
        time.sleep(GREETING_DELAY)
        print("🔵 Воспроизведение приветствия")
        play_audio(greeting_path)
        print("✅ Приветствие завершено")
        print("⏳ Ожидание затихания эха...")
        time.sleep(0.8)
        mic_paused = False
        print("▶️ Микрофон возобновлён")
        update_activity()
        return True
    except Exception as e:
        print(f"❌ {e}")
        mic_paused = False
        return False

# ============================================================
# MAIN
# ============================================================
def main():
    global transfer_to_staff_mode, spa_transfer_mode, room_service_mode, chars_this_call, verdict_played
    global all_requests, staff_mode_active, waiting_for_response

    print("=" * 60)
    print(f"🎤 Ассистент — Комната {ROOM_NUMBER} ({VOICE_ENGINE.upper()})")
    print("=" * 60)

    print("🔵 Шаг 1: start_microphone()")
    start_microphone()
    time.sleep(0.5)

    print("🔵 Шаг 2: start_music()")
    start_music()
    time.sleep(0.5)

    print(f"🔵 Шаг 3: play_greeting_with_delay()")
    play_greeting_with_delay()

    print("🔵 Шаг 4: start_silence_monitor()")
    start_silence_monitor()

    print("\n🎤 Микрофон активен. Говорите...\n")

    not_understood_count = 0
    waiting_for_response = False

    while True:
        try:
            if verdict_played:
                time.sleep(0.5)
                continue

            if not audio_queue.empty():
                user_text = audio_queue.get()
                print(f"🔵 Из очереди: '{user_text}'")
                update_activity()

                if user_text.lower() in ["יציאה", "להתראות", "exit", "quit"]:
                    speak("להתראות! שיהיה לך יום טוב.", language="he")
                    stop_music()
                    stop_recording_and_save()
                    update_room_chars(ROOM_NUMBER, 0, is_call_end=True)
                    break

                # ⭐ СНАЧАЛА проверяем transfer/spa/room_service, ПОТОМ waiting_for_response
                if transfer_to_staff_mode or spa_transfer_mode or room_service_mode:
                    print("⏳ Анализ ответа...")
                    if spa_transfer_mode:
                        result = ai_detect_intent(user_text, mode="spa_response")
                    elif room_service_mode:
                        result = ai_detect_intent(user_text, mode="room_service_response")
                    else:
                        result = ai_detect_intent(user_text, mode="transfer_response")
                    answer = result.get("answer", "other")
                    print(f"📋 AI ответ: {answer}")
                    if answer == "transfer":
                        staff_mode_active = True
                        waiting_for_response = False
                        if spa_transfer_mode:
                            play_spa_transfer_complete()
                        elif room_service_mode:
                            room_service_mode = False
                            play_transfer_complete()
                        else:
                            play_transfer_complete()
                        continue
                    elif answer == "end":
                        print("👤 Отказ")
                        transfer_to_staff_mode = False
                        spa_transfer_mode = False
                        room_service_mode = False
                        staff_mode_active = False
                        waiting_for_response = False
                        continue_path = get_audio_path("continue_conversation")
                        if continue_path:
                            play_audio(continue_path)
                            update_activity()
                        continue
                    else:
                        print("🔄 Другое — выход из режима")
                        transfer_to_staff_mode = False
                        spa_transfer_mode = False
                        room_service_mode = False
                        staff_mode_active = False
                        waiting_for_response = False

                # ⭐ Финальный вердикт при "כן"/"לא"
                if waiting_for_response and (is_yes(user_text) or is_no(user_text)):
                    print(f"🎯 Ответ '{user_text}' — финальный вердикт")
                    if all_requests:
                        requests_text = ", ".join(all_requests)
                    else:
                        requests_text = "הבקשה שלכם"
                    final_reply = f"העברתי את בקשתכם לצוות: {requests_text}. שיהיה לכם חופשה מהנה!"
                    speak(final_reply, language="he")
                    waiting_for_response = False
                    continue

                print("⏳ AI определяет намерение...")
                result = ai_detect_intent(user_text, mode="main")
                intent = result.get("intent", "request")
                topic = result.get("topic")
                print(f"📋 Намерение: {intent}, Тема: {topic}")

                if intent == "transfer":
                    print("👤 Перевод")
                    transfer_to_staff_mode = True
                    waiting_for_response = False   # ⭐ СБРОС
                    confirm_path = get_audio_path("transfer_confirm")
                    if confirm_path:
                        play_audio(confirm_path)
                        update_activity()
                    else:
                        speak("הבנתי. תרצו שאעביר אתכם לנציג אנושי?", language="he")
                    continue

                if intent == "info" and topic:
                    if topic == "spa_transfer":
                        audio_path = get_audio_path("info_spa_transfer")
                        if audio_path:
                            play_audio(audio_path)
                            spa_transfer_mode = True
                            waiting_for_response = False
                            update_activity()
                            continue
                    if topic == "room_service":
                        audio_path = get_audio_path("info_room_service")
                        if audio_path:
                            play_audio(audio_path)
                            room_service_mode = True
                            waiting_for_response = False
                            update_activity()
                            continue
                    if topic == "wifi":
                        audio_path = get_audio_path("info_wifi")
                        if audio_path:
                            play_audio(audio_path)
                            update_activity()
                            continue
                    audio_key = f"info_{topic}"
                    audio_path = get_audio_path(audio_key)
                    if audio_path:
                        play_audio(audio_path)
                        update_activity()
                        continue

                print("⏳ Обработка просьбы...")
                response, not_understood, requests_list = ask_ai_with_memory(user_text, language="he")

                if requests_list:
                    for req in requests_list:
                        if req and req not in all_requests:
                            all_requests.append(req)
                            print(f"📝 Добавлено в просьбы: '{req}'")

                if not_understood:
                    not_understood_count += 1
                    if not_understood_count == 1:
                        audio_path = get_audio_path("not_understood")
                        if audio_path:
                            mic_paused = True
                            time.sleep(0.15)
                            play_audio(audio_path)
                            time.sleep(0.5)
                            mic_paused = False
                            update_activity()
                        else:
                            speak("מצטער, לא הצלחתי להבין.", language="he")
                    else:
                        error_path = get_audio_path("error")
                        if error_path:
                            mic_paused = True
                            time.sleep(0.15)
                            play_audio(error_path)
                            try:
                                stop_recording_and_save()
                            except:
                                pass
                            update_room_chars(ROOM_NUMBER, 0, is_call_end=True)
                            stop_music()
                            stop_ding_loop()
                            os._exit(0)
                else:
                    not_understood_count = 0
                    if "שיהיה לכם חופשה מהנה" in response:
                        speak(response, language="he")
                    else:
                        if "האם אתם רוצים עוד משהו" in response:
                            waiting_for_response = True
                        speak(response, language="he")

            time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n👋 Завершение...")
            stop_music()
            stop_ding_loop()
            stop_recording_and_save()
            break
        except Exception as e:
            print(f"❌ Ошибка main: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

if __name__ == "__main__":
    main()
