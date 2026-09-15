# voice_assistant_v2.py
import os
import sys

# В обычном cmd/PowerShell Windows кодировка может быть cp1251. В консоли
# ассистента есть emoji и иврит, поэтому фиксируем UTF-8 ещё до первых print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# ⭐ CUDA DLL для CTranslate2 (faster-whisper)
SITE_PACKAGES = r"C:\Users\david\AppData\Roaming\Python\Python312\site-packages"
for _p in [
    os.path.join(SITE_PACKAGES, "ctranslate2"),
    os.path.join(SITE_PACKAGES, "nvidia", "cublas", "bin"),
    os.path.join(SITE_PACKAGES, "nvidia", "cuda_runtime", "bin"),
]:
    if os.path.exists(_p):
        try:
            os.add_dll_directory(_p)
        except Exception:
            pass

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

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
from dataclasses import dataclass, field
import random
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

import pygame
import numpy as np
import speech_recognition as sr
import edge_tts

# ============================================================
# FASTER-WHISPER
# ============================================================
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    print("❌ faster-whisper не установлен! pip install faster-whisper")
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
WHATSAPP_PROJECT_DIR = HERE.parent / "Project"
if WHATSAPP_PROJECT_DIR.exists():
    sys.path.insert(0, str(WHATSAPP_PROJECT_DIR))
from queue_utils import append_queue_item
LOADING_SOUND = HERE / "phone-beeps.mp3"
ROOM_STATUS_FILE = HERE / "room_status.json"

# ============================================================
# НАСТРОЙКИ
# ============================================================
MAX_CHARS_PER_CALL = 250
MAX_CHARS_PER_DAY = 5000
KOKORO_COOLDOWN_HOURS = 6
HOTEL_STATUS_KEY = "__hotel_elevenlabs__"
STARTUP_ELEVENLABS_STOP_REASON = None
# После фактического отказа ElevenLabs не пытаемся снова синтезировать,
# пока в аккаунте не хватит символов на полный максимально допустимый ответ.
ELEVENLABS_MIN_CHARS_TO_RESUME = MAX_CHARS_PER_CALL
SILENCE_INTERVAL = 30
PRE_BUFFER_SIZE = 10
ENERGY_THRESHOLD = 700
SILENCE_THRESHOLD = 12
MIN_TEXT_LENGTH = 2
SPEAK_PAUSE_BEFORE = 0.15
SPEAK_PAUSE_AFTER = 0.5
ECHO_PROTECTION_TIME = 0.8

# Постоянный режим нужен для будущей телефонии: процесс остаётся в памяти
# между последовательными звонками. Обычный запуск сохраняет прежнее
# поведение: после завершения звонка процесс закрывается.
CONTINUOUS_SERVICE_MODE = "--continuous" in sys.argv

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
JESSICA_VOICE_ID = "cgSgspJ2msm6clMCkdW9"
ELEVENLABS_MODEL = "eleven_v3"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "groq/compound-mini"

WHISPER_MODEL_NAME = "ivrit-ai/whisper-large-v3-turbo-ct2"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "int8_float16"

YES_WORDS = ["כן", "נכון", "בסדר", "אוקיי", "אוקי", "בטח", "יאללה"]
NO_WORDS = ["לא", "לא תודה", "לא צריך", "לא, תודה", "לא רוצה", "לא נכון"]
# Короткие естественные варианты «больше ничего не нужно». Это именно
# законченные фразы: если после "לא" есть новая просьба, они сюда не попадут.
NATURAL_FINISH_PHRASES = {
    "זהו", "זהו תודה", "זה הכל", "זה הכול", "זה הכל תודה", "זה הכול תודה",
    "תודה זהו", "תודה רבה זהו", "תודה זה הכל", "תודה זה הכול",
    "לא צריך יותר", "לא צריכים יותר", "אין צורך יותר", "אין צורך בעוד משהו",
    "סיימנו", "זה מספיק", "מספיק תודה",
}

# ============================================================
# ЗАГРУЗКА WHISPER
# ============================================================
whisper_model = None

def load_whisper_model():
    global whisper_model, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
    if not WHISPER_AVAILABLE:
        print("⚠️ faster-whisper недоступен")
        return
    print(f"📥 Загрузка {WHISPER_MODEL_NAME} ({WHISPER_DEVICE}/{WHISPER_COMPUTE_TYPE})...")
    try:
        whisper_model = WhisperModel(
            WHISPER_MODEL_NAME,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            num_workers=2,
        )
        print(f"✅ Whisper загружен ({WHISPER_DEVICE}/{WHISPER_COMPUTE_TYPE})")
        return
    except Exception as e:
        print(f"⚠️ Ошибка CUDA: {e}")
    try:
        WHISPER_COMPUTE_TYPE = "float16"
        whisper_model = WhisperModel(WHISPER_MODEL_NAME, device="cuda", compute_type="float16")
        print(f"✅ Whisper загружен (cuda/float16)")
        return
    except Exception as e:
        print(f"⚠️ Ошибка float16: {e}")
    try:
        WHISPER_DEVICE = "cpu"
        WHISPER_COMPUTE_TYPE = "int8"
        whisper_model = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
        print(f"✅ Whisper загружен (cpu/int8)")
    except Exception as e:
        print(f"❌ Whisper не загрузился: {e}")
        whisper_model = None

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
call_end_event = threading.Event()
static_sound_cache = {}


@dataclass
class CallSession:
    """Изолированное состояние одного гостя.

    Сейчас один процесс обслуживает один аудиоканал за раз. Объект отделяет
    историю и список просьб от моделей, чтобы следующий этап — SIP-сессии с
    несколькими параллельными звонками — не смешивал данные гостей.
    """
    room_number: str
    voice_engine: str
    started_at: float = field(default_factory=time.time)
    conversation_history: list = field(default_factory=list)
    all_requests: list = field(default_factory=list)


active_session = None


def preload_static_audio():
    """Кэширует постоянные MP3, не затрагивая временные TTS-файлы."""
    if not CONTINUOUS_SERVICE_MODE:
        return
    loaded = 0
    for audio_dir in (HERE / "ElevenLabs_RECORD", HERE / "Kokoro_RECORD"):
        if not audio_dir.exists():
            continue
        for path in audio_dir.glob("*.mp3"):
            key = str(path.resolve())
            if key in static_sound_cache:
                continue
            try:
                static_sound_cache[key] = pygame.mixer.Sound(key)
                loaded += 1
            except Exception as e:
                print(f"⚠️ Не удалось предзагрузить {path.name}: {e}")
    print(f"⚡ Предзагружено записей: {loaded}")

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
    try:
        ding_channel.stop()
    except Exception:
        pass

# ============================================================
# play_audio
# ============================================================
def play_audio(path, volume=1.0):
    stop_ding_loop()
    try:
        if not Path(path).exists():
            print(f"❌ Не найден: {path}")
            return False
        sound_path = str(Path(path).resolve())
        sound = static_sound_cache.get(sound_path)
        if sound is None:
            sound = pygame.mixer.Sound(sound_path)
            # В постоянном режиме кэшируем только записи проекта. Временные
            # файлы ElevenLabs/Kokoro нельзя сохранять: они удаляются после
            # воспроизведения.
            if CONTINUOUS_SERVICE_MODE and sound_path.startswith(str(HERE.resolve())):
                static_sound_cache[sound_path] = sound
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
# ФИЛЬТР ГАЛЛЮЦИНАЦИЙ
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


def get_hotel_elevenlabs_status(status):
    """Общий счётчик ElevenLabs для всего отеля; Kokoro сюда не попадает."""
    today = get_today_str()
    hotel = status.get(HOTEL_STATUS_KEY)
    if hotel is None:
        # Миграция старого room_status.json: суммируем уже потраченные сегодня
        # ElevenLabs-символы по комнатам один раз, прежде чем вести общий счётчик.
        hotel = {
            "chars_today": sum(
                item.get("chars_today", 0)
                for key, item in status.items()
                if key != HOTEL_STATUS_KEY
                and isinstance(item, dict)
                and item.get("last_reset_day") == today
            ),
            "last_reset_day": today,
        }
    if hotel.get("last_reset_day") != today:
        hotel["chars_today"] = 0
        hotel["last_reset_day"] = today
        hotel.pop("kokoro_for_day", None)
    status[HOTEL_STATUS_KEY] = hotel
    return hotel


def set_hotel_kokoro_for_today():
    status = load_room_status()
    hotel = get_hotel_elevenlabs_status(status)
    hotel["kokoro_for_day"] = get_today_str()
    status[HOTEL_STATUS_KEY] = hotel
    save_room_status(status)


def set_elevenlabs_tokens_depleted():
    status = load_room_status()
    hotel = get_hotel_elevenlabs_status(status)
    hotel["tokens_depleted"] = True
    status[HOTEL_STATUS_KEY] = hotel
    save_room_status(status)


def clear_elevenlabs_tokens_depleted(status):
    hotel = get_hotel_elevenlabs_status(status)
    if hotel.pop("tokens_depleted", None):
        status[HOTEL_STATUS_KEY] = hotel
        save_room_status(status)

def elevenlabs_remaining_chars():
    """Остаток ElevenLabs; None, если API баланса временно недоступен."""
    try:
        response = requests.get(
            "https://api.elevenlabs.io/v1/user/subscription",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            timeout=10,
        )
        if response.status_code != 200:
            print(f"⚠️ ElevenLabs: не удалось проверить баланс ({response.status_code})")
            return None
        subscription = response.json()
        used = subscription.get("character_count")
        limit = subscription.get("character_limit")
        if used is None or limit is None:
            print("⚠️ ElevenLabs: баланс не указан в ответе API")
            return None
        remaining = max(0, limit - used)
        print(f"💰 ElevenLabs: осталось {remaining}/{limit} символов")
        return remaining
    except Exception as e:
        print(f"⚠️ ElevenLabs: проверка баланса недоступна: {e}")
        return None


def elevenlabs_has_tokens():
    """Совместимая проверка: True/False/None без раскрытия детали баланса."""
    remaining = elevenlabs_remaining_chars()
    if remaining is None:
        return None
    return remaining > 0

def get_room_engine(room_number):
    global STARTUP_ELEVENLABS_STOP_REASON
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
    hotel = get_hotel_elevenlabs_status(status)
    if hotel.get("kokoro_for_day") == today or hotel.get("chars_today", 0) >= MAX_CHARS_PER_DAY:
        print(f"🎤 Отель: ЛИМИТ {MAX_CHARS_PER_DAY}/день → KOKORO до конца дня")
        return 'kokoro'

    remaining_chars = elevenlabs_remaining_chars()
    # Ошибка синтеза важнее округлённого/запаздывающего остатка API.
    # Пока нет запаса на один полный ответ, новые звонки сразу идут в Kokoro.
    if hotel.get("tokens_depleted"):
        if remaining_chars is not None and remaining_chars >= ELEVENLABS_MIN_CHARS_TO_RESUME:
            clear_elevenlabs_tokens_depleted(status)
            print("🎤 ElevenLabs: баланс восстановлен → ELEVENLABS")
        else:
            shown_remaining = "неизвестен" if remaining_chars is None else remaining_chars
            print(
                f"🎤 ElevenLabs: после ошибки синтеза остаток {shown_remaining} "
                f"(< {ELEVENLABS_MIN_CHARS_TO_RESUME}) → KOKORO"
            )
            return 'kokoro'

    if remaining_chars == 0:
        if not hotel.get("tokens_depleted"):
            hotel["tokens_depleted"] = True
            status[HOTEL_STATUS_KEY] = hotel
            save_room_status(status)
            STARTUP_ELEVENLABS_STOP_REASON = "В ELEVENLABS НЕТ ТОКЕНОВ"
            print("🎤 ElevenLabs: токены закончились → текущий звонок будет завершён")
        else:
            print("🎤 ElevenLabs: токенов нет → KOKORO")
        return 'kokoro'
    print(f"🎤 Комната {room_number}: ELEVENLABS ({room.get('chars_today', 0)}/{MAX_CHARS_PER_DAY}), отель {hotel.get('chars_today', 0)}/{MAX_CHARS_PER_DAY}")
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
    # Эта функция вызывается только после успешной озвучки ElevenLabs.
    # Kokoro не расходует и не увеличивает лимиты.
    if chars > 0:
        hotel = get_hotel_elevenlabs_status(status)
        hotel["chars_today"] = hotel.get("chars_today", 0) + chars
        status[HOTEL_STATUS_KEY] = hotel
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

if VOICE_ENGINE == 'kokoro' or CONTINUOUS_SERVICE_MODE:
    if CONTINUOUS_SERVICE_MODE and VOICE_ENGINE != 'kokoro':
        print("⚡ Постоянный режим: заранее загружаю Kokoro для мгновенного переключения")
    print("📥 Загрузка Kokoro...")
    try:
        import soundfile as sf
        from kokoro import KModel
        import torch
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

def ensure_kokoro_loaded():
    """Loads Kokoro if ElevenLabs became unavailable during an active call."""
    global km, voice, phonemize_hebrew, g2p
    if km is not None and voice is not None:
        return True
    try:
        import torch
        from kokoro import KModel
        from hebrew_g2p import phonemize_hebrew as ph
        km = KModel(repo_id="hexgrad/Kokoro-82M", model=str(KOKORO_MODEL_PATH)).to("cpu").eval()
        voice = torch.load(KOKORO_VOICE_PATH, map_location="cpu", weights_only=True)
        phonemize_hebrew = ph
        try:
            from renikud_onnx import G2P
            g2p = G2P(str(HERE / "renikud_model.onnx"))
        except Exception as e:
            print(f"⚠️ ReNikud: {e}")
            g2p = None
        print("✅ Kokoro загружен для переключения")
        return True
    except Exception as e:
        print(f"❌ Не удалось загрузить Kokoro: {e}")
        return False


def configure_call(room_number):
    """Выбирает движок и аудиозаписи для следующей последовательной сессии."""
    global ROOM_NUMBER, VOICE_ENGINE, AUDIO_DIR, VOICE_SUFFIX
    global STARTUP_ELEVENLABS_STOP_REASON, active_session

    ROOM_NUMBER = room_number
    STARTUP_ELEVENLABS_STOP_REASON = None
    VOICE_ENGINE = get_room_engine(ROOM_NUMBER)
    if VOICE_ENGINE == "kokoro":
        AUDIO_DIR = HERE / "Kokoro_RECORD"
        VOICE_SUFFIX = "_kokoro"
        ensure_kokoro_loaded()
    else:
        AUDIO_DIR = HERE / "ElevenLabs_RECORD"
        VOICE_SUFFIX = ""
    active_session = CallSession(ROOM_NUMBER, VOICE_ENGINE)


def reset_call_runtime_state():
    """Очищает данные прежнего гостя, оставляя Whisper/Kokoro в памяти."""
    global audio_queue, is_speaking, microphone_active, mic_thread, stop_mic, mic_paused
    global conversation_history, stop_loading_ding, ding_thread_running, recording_frames
    global is_recording, current_conversation_text, transfer_to_staff_mode, spa_transfer_mode
    global room_service_mode, staff_mode_active, chars_this_call, verdict_played, all_requests
    global last_activity_time, stop_silence_monitor, silence_thread, active_session

    stop_ding_loop()
    stop_silence_monitor = True
    stop_mic = False
    mic_paused = False
    is_speaking = False
    microphone_active = False
    mic_thread = None
    audio_queue = queue.Queue()
    recording_frames = []
    is_recording = False
    current_conversation_text = []
    transfer_to_staff_mode = False
    spa_transfer_mode = False
    room_service_mode = False
    staff_mode_active = False
    chars_this_call = 0
    verdict_played = False
    last_activity_time = time.time()
    silence_thread = None
    call_end_event.clear()
    if active_session is None:
        active_session = CallSession(ROOM_NUMBER, VOICE_ENGINE)
    conversation_history = active_session.conversation_history
    all_requests = active_session.all_requests


def stop_call_workers():
    """Освобождает аудиовход перед следующей сессией в постоянном режиме."""
    global stop_mic, stop_silence_monitor, mic_paused
    stop_mic = True
    stop_silence_monitor = True
    mic_paused = True
    stop_ding_loop()
    stop_music()
    if mic_thread and mic_thread.is_alive():
        mic_thread.join(timeout=3)
    if silence_thread and silence_thread.is_alive():
        silence_thread.join(timeout=2)
    try:
        voice_channel.stop()
    except Exception:
        pass


def finish_current_call(exit_code=0):
    """Завершает сессию; в --continuous возвращает управление диспетчеру."""
    call_end_event.set()
    if not CONTINUOUS_SERVICE_MODE:
        os._exit(exit_code)

def switch_to_kokoro(reason):
    global VOICE_ENGINE
    print(f"🎤 {reason} → KOKORO + stop_request")
    stop_path = get_audio_path("stop_request_and_changing_line")
    if stop_path:
        play_audio(stop_path)
    set_kokoro_cooldown(ROOM_NUMBER)
    VOICE_ENGINE = 'kokoro'
    ensure_kokoro_loaded()


def end_call_after_elevenlabs_limit(reason, room_cooldown=False, hotel_daily_limit=False, tokens_depleted=False):
    """Останавливает текущий звонок из-за лимита ElevenLabs.

    Kokoro не является причиной этого пути: он не имеет лимитов и не меняет
    счётчики. При лимите за звонок только следующая сессия этой комнаты будет
    Kokoro шесть часов; при дневном лимите Kokoro включается всему отелю.
    """
    print("=" * 60)
    print(f"⚠️ ELEVENLABS: {reason}")
    print("доделать логику когда программа будет готова")
    print("=" * 60)
    if room_cooldown:
        set_kokoro_cooldown(ROOM_NUMBER)
    if hotel_daily_limit:
        set_hotel_kokoro_for_today()
    if tokens_depleted:
        set_elevenlabs_tokens_depleted()

    stop_ding_loop()
    stop_path = get_audio_path("stop_request_and_changing_line", engine='eleven')
    if stop_path:
        play_audio(stop_path)
    else:
        print("❌ Не найден stop_request_and_changing_line.mp3")

    try:
        stop_recording_and_save()
    except Exception:
        pass
    update_room_chars(ROOM_NUMBER, 0, is_call_end=True)
    stop_music()
    stop_ding_loop()
    time.sleep(1)
    return finish_current_call()

def elevenlabs_quota_error(response):
    try:
        details = response.text.lower()
    except Exception:
        details = ""
    quota_words = ("quota", "credit", "character", "insufficient", "subscription", "payment")
    return response.status_code in (402, 429) or any(word in details for word in quota_words)

def exceeds_daily_limit(chars_to_use):
    status = load_room_status()
    hotel = get_hotel_elevenlabs_status(status)
    return hotel.get("chars_today", 0) + chars_to_use > MAX_CHARS_PER_DAY

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
    "transfer_answer_required": "transfer_answer_required",
    "ai_unavailable_transfer": "ai_unavailable_transfer",
    "separate_request_or_question": "separate_request_or_question",
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
        path = (
            HERE / "Kokoro_RECORD" / "stop_request_and_changing_line_kokoro.mp3"
            if engine == "kokoro"
            else HERE / "ElevenLabs_RECORD" / "stop_request_and_changing_line.mp3"
        )
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


def is_wifi_related(text):
    """Internet and Wi-Fi issues always use the recorded Wi-Fi answer, not AI routing."""
    normalized = text.lower().replace("-", " ")
    wifi_markers = (
        "אינטרנט", "אינטר", "וויפי", "ווי פיי", "וויי פיי",
        "wifi", "wi fi", "wi-fi",
    )
    return any(marker in normalized for marker in wifi_markers)


def is_urgent_request(text, intent=None):
    """Страховка для критических фраз, даже если классификатор дал сбой.

    Это именно чрезвычайные случаи: они не идут в обычный цикл накопления
    просьб и не ждут финального вердикта.
    """
    if intent == "urgent":
        return True
    normalized = text.lower()
    urgent_patterns = (
        ("מעשן", "מסדרון"),             # курят в коридоре
        ("אדם זר",),                     # незнакомец у комнаты
        ("מרגיש לא טוב",),               # гостю плохо
        ("מים", "חשמל"),                # вода рядом с электричеством
        ("צועקים", "מפחד"),             # крики, гость боится
        ("ריח שרוף", "שקע"),            # запах гари у розетки
        ("ילד", "ננעל"),                # ребёнок заперт
        ("דופק", "לא מזדהה"),           # неизвестный стучит в дверь
    )
    return any(all(marker in normalized for marker in pattern) for pattern in urgent_patterns)


def is_complaint_request(text, intent=None):
    """Страховка для явных жалоб, которые сразу передаются персоналу."""
    if intent == "complaint":
        return True
    normalized = text.lower()
    complaint_patterns = (
        ("יש לי תלונה",),
        ("לא רצינו ניקיון",),
        ("נכנסו", "לחדר", "ניקיון"),
        ("כל הבעיות", "מנהל"),
    )
    return any(all(marker in normalized for marker in pattern) for pattern in complaint_patterns)

# ============================================================
# WHATSAPP
# ============================================================
QUEUE_FILE = Path(r"C:\Users\david\Desktop\WhatsApp\Project\send_queue.json")


def enqueue_whatsapp_message(item, label):
    """Добавляет сообщение без гонки с другим звонком или монитором."""
    try:
        queue_size = append_queue_item(QUEUE_FILE, item)
        print(f"{label} (всего: {queue_size})")
        return True
    except Exception as e:
        print(f"❌ Ошибка WhatsApp-очереди: {e}")
        return False

def send_verdict_to_whatsapp(requests_text, audio_file_path=None):
    print("🔍 send_verdict_to_whatsapp!")
    if not requests_text:
        requests_text = "בקשה לא זוהתה"
    message = f"""🎤 *הודעה מהמערכת הדיגיטלית* (AI)

📋 *האורח מ-{ROOM_NUMBER} ביקש:*
{requests_text}"""
    return enqueue_whatsapp_message({
        "id": str(int(time.time() * 1000)),
        "text": message,
        "audio_file": audio_file_path,
        "created_at": time.time()
    }, "📤 В очередь")


def send_urgent_to_whatsapp(urgent_text, audio_file_path=None):
    """Немедленно ставит чрезвычайный случай в главную очередь WhatsApp."""
    message = f"""🎤 *הודעה מהמערכת הדיגיטלית* (AI)

🚨 *דחוף — העברה מיידית לצוות*

🏨 *חדר:* {ROOM_NUMBER}
📋 *דיווח האורח:*
{urgent_text}"""
    return enqueue_whatsapp_message({
        "id": str(int(time.time() * 1000)),
        "text": message,
        "audio_file": audio_file_path,
        "created_at": time.time(),
        "kind": "urgent",
    }, "🚨 В WhatsApp-очередь URGENT")


def send_complaint_to_whatsapp(complaint_text, audio_file_path=None):
    """Ставит жалобу гостя в главную очередь WhatsApp для персонала."""
    message = f"""🎤 *הודעה מהמערכת הדיגיטלית* (AI)

⚠️ *תלונת אורח — להעביר לצוות*

🏨 *חדר:* {ROOM_NUMBER}
📋 *דיווח האורח:*
{complaint_text}"""
    return enqueue_whatsapp_message({
        "id": str(int(time.time() * 1000)),
        "text": message,
        "audio_file": audio_file_path,
        "created_at": time.time(),
        "kind": "complaint",
    }, "⚠️ В WhatsApp-очередь COMPLAINT")

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
                    finish_current_call()
                    return
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
    clean = re.sub(r"[.!?,;:]+", " ", text.strip().lower())
    clean = " ".join(clean.split())
    return clean in YES_WORDS

def is_no(text):
    clean = re.sub(r"[.!?,;:]+", " ", text.strip().lower())
    clean = " ".join(clean.split())
    normalized_no_words = {
        " ".join(re.sub(r"[.!?,;:]+", " ", word.lower()).split())
        for word in NO_WORDS
    }
    return clean in normalized_no_words


def is_natural_finish(text):
    """Безопасно завершает только самостоятельные, законченные ответы гостя."""
    clean = re.sub(r"[.!?,;:]+", " ", text.strip().lower())
    clean = " ".join(clean.split())
    return clean in NATURAL_FINISH_PHRASES

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
                if whisper_model is not None:
                    try:
                        print(f"🎤 Whisper ivrit...")
                        t0 = time.time()
                        segments, _ = whisper_model.transcribe(
                            wav_path,
                            language="he",
                            beam_size=1,
                            best_of=1,
                            temperature=0.0,
                            condition_on_previous_text=False,
                            without_timestamps=True,
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=300),
                            word_timestamps=False,
                        )
                        text = " ".join(seg.text for seg in segments).strip()
                        dt = time.time() - t0
                        print(f"🔍 Whisper ({dt:.2f}s): '{text}'")
                        text = filter_whisper_hallucinations(text)
                        if not text:
                            print("🚫 Отфильтрован")
                    except Exception as e:
                        print(f"⚠️ Whisper ошибка: {e}")
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
                        # Гость уже распознан. Не оставляем «динь-динь» на
                        # время ответа DeepSeek/Groq: это не ускоряет ИИ и
                        # только создаёт лишнее ожидание для гостя.
                        stop_ding_loop()
                        print("🔕 ДИНЬ СТОП — речь распознана, анализ ИИ")
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
        import torch
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
    global chars_this_call, VOICE_ENGINE
    if not text or len(text.strip()) < 2:
        print(f"⚠️ Пустой текст")
        return False, 0
    try:
        clean_text = text.replace('"', '').replace("'", "").replace("\n", " ").strip()
        chars_to_use = len(clean_text)
        if chars_this_call + chars_to_use > MAX_CHARS_PER_CALL:
            print(f"⚠️ Превышение за звонок: {chars_this_call} + {chars_to_use} > {MAX_CHARS_PER_CALL}")
            end_call_after_elevenlabs_limit(
                f"ЛИМИТ {MAX_CHARS_PER_CALL} СИМВОЛОВ ЗА ЗВОНОК",
                room_cooldown=True,
            )
        if exceeds_daily_limit(chars_to_use):
            end_call_after_elevenlabs_limit(
                f"ЛИМИТ {MAX_CHARS_PER_DAY} СИМВОЛОВ В ДЕНЬ",
                hotel_daily_limit=True,
            )
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{JESSICA_VOICE_ID}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept-Charset": "utf-8"
        }
        data = {
            "text": clean_text,
            "model_id": ELEVENLABS_MODEL,
            "language_code": "he",
            "voice_settings": {
                "stability": 0.4,
                "similarity_boost": 0.8,
                "style": 0.5,
                "use_speaker_boost": True
            }
        }
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
            if elevenlabs_quota_error(response):
                end_call_after_elevenlabs_limit(
                    f"ElevenLabs: закончились токены ({response.status_code})",
                    tokens_depleted=True,
                )
            else:
                print(f"❌ ElevenLabs: ошибка {response.status_code} → KOKORO")
                VOICE_ENGINE = 'kokoro'
                ensure_kokoro_loaded()
            return speak_kokoro(text), 0
    except Exception as e:
        print(f"❌ ElevenLabs: {e} → KOKORO")
        VOICE_ENGINE = 'kokoro'
        ensure_kokoro_loaded()
        return speak_kokoro(text), 0

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
    end_call_after_elevenlabs_limit(
        f"ЛИМИТ {MAX_CHARS_PER_CALL} СИМВОЛОВ ЗА ЗВОНОК",
        room_cooldown=True,
    )

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
    return finish_current_call()

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
        speak_elevenlabs(text, language)
    time.sleep(SPEAK_PAUSE_AFTER)
    is_speaking = False
    mic_paused = False
    print("🎤 Микрофон возобновлён")
    update_activity()

# ============================================================
# AI С ПАМЯТЬЮ
# ============================================================
class AIProvidersUnavailable(Exception):
    """Ни DeepSeek, ни резервный Groq не смогли ответить."""


def create_ai_completion(messages, temperature, max_tokens):
    """Запрашивает DeepSeek и без задержек переключается на Groq при сбое."""
    from openai import OpenAI

    providers = [
        ("DeepSeek", DEEPSEEK_API_KEY, DEEPSEEK_URL, DEEPSEEK_MODEL),
        ("Groq", GROQ_API_KEY, GROQ_URL, GROQ_MODEL),
    ]
    errors = []
    for provider_name, api_key, base_url, model in providers:
        if not api_key:
            errors.append(f"{provider_name}: ключ отсутствует")
            continue
        try:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=10.0,
                max_retries=0,
            )
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            print(f"✅ AI: {provider_name}")
            return response
        except Exception as e:
            print(f"⚠️ {provider_name} недоступен: {e}")
            errors.append(f"{provider_name}: {e}")

    raise AIProvidersUnavailable("; ".join(errors))


def ask_ai_with_memory(prompt, language="he"):
    global conversation_history
    print("=" * 60)
    print("🔍 ДИАГНОСТИКА AI")
    print("=" * 60)
    print(f"📝 prompt: '{prompt}'")
    print(f"📚 История: {len(conversation_history)}")
    print(f"📋 Уже собрано просьб: {all_requests}")

    current_requests_text = ", ".join(all_requests) if all_requests else "אין"
    elevenlabs_female_voice_instruction = ""
    if VOICE_ENGINE == "eleven":
        elevenlabs_female_voice_instruction = """
⚠️ הקול שמדבר את התשובה הוא קול נשי. כשאת מדברת על עצמך, השתמשי תמיד בלשון נקבה:
"אני מבינה", "אני מצטערת", "אני שמחה", "אעביר". לעולם אל תכתבי "אני מבין" או צורת זכר אחרת על עצמך.
אל האורחים המשיכי לפנות בלשון רבים: לכם / אתם.
"""

    try:
        system_prompt = f"""אתה העוזר הדיגיטלי של מלון רויאל ים המלח.

⚠️ תמיד תחזיר תשובת JSON בלבד!

{{
  "understood": true,
  "reply": "הטקסט שתרצה לומר לאורח",
  "requests": ["מגבות", "שמפו"],
  "wait_for_more": true,
  "dialogue_action": "request"
}}

או אם לא הבנת:
{{
  "understood": false,
  "reply": "סלחו לי, אשמח להבין בדיוק מה אתם צריכים.",
  "requests": [],
  "wait_for_more": false,
  "dialogue_action": "other"
}}

📌 בקשות שכבר נאספו בשיחה: {current_requests_text}

{elevenlabs_female_voice_instruction}

📌 הכללים:

1. אתה זוכר מה האורח ביקש קודם — все בקשות уже в списке выше.
2. Если гость добавляет НОВУЮ просьбу — добавь её в "requests".
3. Если гость повторяет — не дублируй.
4. Если гость говорит "כן"/"לא"/"תודה" — "requests": [].
5. Если не понял — understood: false.
6. После 2 непонятых — "מצטער. לא הצלחתי להבין."
7. בבקשה רגילה שהובנה: תמיד "wait_for_more": true.
8. "wait_for_more": false רק אם לא הבנת או שאין להמשיך את השיחה.

📌 "dialogue_action" מתאר את משמעות התגובה בהקשר של הבקשות שכבר נאספו:
- "finish" — האורח אומר שאין לו עוד בקשות: למשל "זהו, תודה", "זה הכל", "לא צריך יותר", "סיימנו". requests חייב להיות [].
- "continue" — האורח מאשר שיש לו עוד מה לבקש, אך עדיין לא אמר בקשה חדשה: למשל "כן", "נכון", "בסדר". requests חייב להיות [].
- "request" — האורח מוסיף בקשה חדשה. גם "לא, אני צריך עוד מגבות" הוא request ולא finish.
- "other" — תשובה שאינה ברורה או אינה תשובה לשאלה אם יש עוד בקשות.
אם לא נאספו בקשות קודמות, אל תחזיר finish או continue.

⚠️⚠️⚠️ שדה "requests" — רשימת בקשות **נקיות** בעברית תקנית:
- "אפשר מגבות לחדר" → ["מגבות לחדר"]
- "אני צריך מרכך שמפו ומגבות" → ["מרכך", "שמפו", "מגבות"]
- "לא תודה" → []
- "כן" → []
- "מתי ארוחת בוקר" → []

⚠️⚠️⚠️ פסק הדין הסופי נבנה על ידי התוכנית רק אחרי שהאורח אמר "לא" או "לא תודה".
אל תכתבי את פסק הדין הסופי בעצמך.

📝 תבניות:

🔹 Первый ответ:
"בשמחה נשלח לכם [רשימת בקשות]. האם אתם רוצים עוד משהו?"

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

        response = create_ai_completion(messages, temperature=0.3, max_tokens=120)

        finish_reason = response.choices[0].finish_reason
        usage = response.usage
        raw_content = response.choices[0].message.content
        print(f"✅ finish_reason: {finish_reason}")
        print(f"📊 tokens: {usage.total_tokens}")

        if not raw_content:
            print(f"❌ Пустой content!")
            return "לא הצלחתי להבין, נסו שוב", False, [], False, "other"

        raw_answer = raw_content.strip()
        requests_list = []
        wait_for_more = False
        dialogue_action = "other"
        try:
            parsed = json.loads(raw_answer)
            understood = parsed.get("understood", True)
            reply = parsed.get("reply", raw_answer)
            requests_list = parsed.get("requests", [])
            if not isinstance(requests_list, list):
                requests_list = []
            requests_list = [r.strip() for r in requests_list if isinstance(r, str) and r.strip()]
            raw_wait_for_more = parsed.get("wait_for_more")
            if isinstance(raw_wait_for_more, bool):
                wait_for_more = raw_wait_for_more
            else:
                # Совместимость со старым/неполным JSON: обычная новая просьба
                # всё равно остаётся в цикле до ответа гостя.
                wait_for_more = bool(understood and requests_list)
            candidate_action = parsed.get("dialogue_action", "other")
            if candidate_action in {"finish", "continue", "request", "other"}:
                dialogue_action = candidate_action
            if not reply or len(reply.strip()) < 2:
                reply = "לא הצלחתי להבין, נסו שוב"
        except json.JSONDecodeError:
            understood = True
            reply = raw_answer if len(raw_answer) > 3 else "לא הצלחתי להבין, נסו שוב"
            requests_list = []
            wait_for_more = False

        conversation_history.append({"role": "user", "content": prompt})
        conversation_history.append({"role": "assistant", "content": reply})
        if len(conversation_history) > MAX_HISTORY:
            # Сохраняем тот же список CallSession, а не подменяем его новым.
            del conversation_history[:-MAX_HISTORY]

        not_understood = not understood
        print(f"💬 reply: '{reply}'")
        print(f"📋 requests от AI: {requests_list}")
        print(f"↪️ wait_for_more: {wait_for_more}")
        print(f"↪️ dialogue_action: {dialogue_action}")
        print("=" * 60)

        return reply, not_understood, requests_list, wait_for_more, dialogue_action

    except AIProvidersUnavailable:
        raise
    except Exception as e:
        print(f"❌ Ошибка AI: {e}")
        return "לא הצלחתי להבין, נסו שוב", False, [], False, "other"

# ============================================================
# AI-КЛАССИФИКАТОР
# ============================================================
def ai_detect_intent(user_text, mode="main"):
    try:
        if mode == "main":
            system_prompt = """אתה מסווג בקשות של אורחי מלון בעברית. תן תשובת JSON בלבד.

1. "urgent" - מצב חירום שמחייב העברה מיידית לצוות, ללא המתנה לתשובת האורח:
   - מעשנים במסדרון
   - אדם זר ליד החדר, מישהו דופק בדלת ולא מזדהה
   - אורח מרגיש לא טוב
   - מים ליד חשמל, ריח שרוף משקע
   - קולות צעקה במסדרון והאורח מפחד
   - ילד נעול בחדר

2. "complaint" - תלונה על שירות או על התנהלות במלון, שמחייבת העברה מיידית לצוות:
   - "נכנסו לחדר שלנו בזמן שלא רצינו ניקיון"
   - "יש לי תלונה"
   - "אחרי כל הבעיות בחדר אני רוצה לדבר עם מנהל"

3. "transfer" - האורח רוצה לדבר עם נציג אנושי

4. "info" - שאלה או בקשה על נושאי המלון:
   - breakfast, lunch, dinner, all_meals - ארוחות
   - lobby, pool, beach, synagogue, rules, wifi - מידע כללי
   - spa_transfer - כל דבר עם ספא / עיסוי / מסאז' / טיפול!
     ⚠️⚠️⚠️ מילים ש**תמיד** מובילות ל-spa_transfer:
     "עיסוי", "עיסויים", "מסאז'", "מסאז", "מסיז'", "ספא", "ספה", "טיפול",
     "טיפולים", "להזמין עיסוי", "אני רוצה עיסוי", "אפשר עיסוי",
     "טיפול בספא", "להזמין מסאז'", "מעסה", "ג'קוזי", "ג'קוזי פרטי",
     "בריכה מקורה", "סאונה", "חדר אדים", "טיפול זוגי"
   - room_service - שירות חדרים

5. "separate" - רק אם האורח משלב באותו משפט בקשה לצוות ושאלת מידע אחרת.
   דוגמה: "אני רוצה מגבות ולשאול מתי יש ארוחת בוקר".
   במקרה כזה אין לטפל באף חלק בנפרד; החזר רק {"intent": "separate"}.

6. "request" - כל דבר אחר (מגבות, שמפו, מרכך, תיקון וכו')

⚠️ "כן", "לא", "בסדר", "תודה" - תמיד "request"!

החזר JSON:
{"intent": "urgent"} / {"intent": "complaint"} / {"intent": "transfer"} / {"intent": "info", "topic": "spa_transfer"} / {"intent": "info", "topic": "wifi"} / {"intent": "separate"} / {"intent": "request"}
"""
            user_message = user_text
        elif mode == "transfer_response":
            system_prompt = """הקשר: שאלו "האם אתם רוצים שאעביר אתכם לנציג אנושי?"

"transfer" (הסכמה) כולל: "כן", "נכון", "בטח", "בסדר", "תעבירו אותי", "אני רוצה".
"end" (סירוב) כולל: "לא", "לא נכון", "לא תודה", "לא רוצה", "אין צורך".
כל שאלה או בקשה אחרת היא "other" — גם אם היא שאלה על המלון.

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
        response = create_ai_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=100,
        )
        raw_content = response.choices[0].message.content
        if not raw_content or len(raw_content) < 5:
            if mode == "main":
                return {"intent": "request"}
            else:
                return {"answer": "other"}
        result = json.loads(raw_content)
        return result
    except AIProvidersUnavailable as e:
        print(f"❌ Все AI-провайдеры недоступны: {e}")
        if mode == "main":
            return {"intent": "ai_unavailable"}
        return {"answer": "ai_unavailable"}
    except Exception as e:
        print(f"⚠️ Ошибка классификатора: {e}")
        if mode == "main":
            return {"intent": "request"}
        else:
            return {"answer": "other"}

# ============================================================
# ЗАВЕРШЕНИЯ
# ============================================================
def play_ai_unavailable_transfer():
    """Сообщает о недоступности ИИ и завершает звонок для перевода в מרכזיה."""
    print("☎️ AI недоступен → перевод в מרכזיה")
    stop_ding_loop()
    try:
        unavailable_path = get_audio_path("ai_unavailable_transfer")
        if unavailable_path:
            play_audio(unavailable_path)
        else:
            speak("זמני לא ניתן לדבר איתי אני מעביר אותכם למרכזיה", language="he")
    finally:
        try:
            stop_recording_and_save()
        except Exception:
            pass
        update_room_chars(ROOM_NUMBER, 0, is_call_end=True)
        stop_music()
        stop_ding_loop()
        time.sleep(1)
        return finish_current_call()


def play_urgent_transfer(urgent_text):
    """Передаёт критическое обращение персоналу без TTS и финального вердикта."""
    print("=" * 60)
    print("🚨 URGENT: немедленный перевод к персоналу и WhatsApp")
    print("=" * 60)
    stop_ding_loop()
    audio_path = None
    try:
        audio_path = stop_recording_and_save()
    except Exception as e:
        print(f"⚠️ Не удалось сохранить urgent-запись: {e}")
    send_urgent_to_whatsapp(urgent_text, audio_path)

    stop_path = get_audio_path("stop_request_and_changing_line", engine=VOICE_ENGINE)
    if stop_path:
        play_audio(stop_path)
    else:
        print("❌ Не найден stop_request_and_changing_line.mp3")
    print("доделать логику когда программа будет готова")

    update_room_chars(ROOM_NUMBER, 0, is_call_end=True)
    stop_music()
    stop_ding_loop()
    time.sleep(1)
    return finish_current_call()


def play_complaint_transfer(complaint_text):
    """Передаёт жалобу персоналу без TTS и финального вердикта."""
    print("=" * 60)
    print("⚠️ COMPLAINT: перевод к персоналу и WhatsApp")
    print("=" * 60)
    stop_ding_loop()
    audio_path = None
    try:
        audio_path = stop_recording_and_save()
    except Exception as e:
        print(f"⚠️ Не удалось сохранить complaint-запись: {e}")
    send_complaint_to_whatsapp(complaint_text, audio_path)

    stop_path = get_audio_path("stop_request_and_changing_line", engine=VOICE_ENGINE)
    if stop_path:
        play_audio(stop_path)
    else:
        print("❌ Не найден stop_request_and_changing_line.mp3")
    print("доделать логику когда программа будет готова")

    update_room_chars(ROOM_NUMBER, 0, is_call_end=True)
    stop_music()
    stop_ding_loop()
    time.sleep(1)
    return finish_current_call()


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
        return finish_current_call()
    except Exception as e:
        print(f"❌ {e}")
        return finish_current_call(exit_code=1)

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
        return finish_current_call()
    except Exception as e:
        print(f"❌ {e}")
        return finish_current_call(exit_code=1)

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
    global all_requests, staff_mode_active, waiting_for_response, mic_paused

    if STARTUP_ELEVENLABS_STOP_REASON:
        end_call_after_elevenlabs_limit(
            STARTUP_ELEVENLABS_STOP_REASON,
            tokens_depleted=True,
        )
        if call_end_event.is_set():
            return True

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
            if call_end_event.is_set():
                print("☎️ Сессия завершена — возвращаю управление диспетчеру")
                return True
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
                    if answer == "ai_unavailable":
                        play_ai_unavailable_transfer()
                        continue
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
                        if transfer_to_staff_mode:
                            print("🔄 Transfer: нужен ответ да или нет")
                            required_path = get_audio_path("transfer_answer_required")
                            if required_path:
                                play_audio(required_path)
                            else:
                                speak("לפני שאני אוכל לעזור לכם בשאלות אחרות, נא תגידו אם אתם מעוניינים לדבר עם נציג אנושי.")
                            update_activity()
                            continue
                        print("🔄 Другое — выход из режима")
                        spa_transfer_mode = False
                        room_service_mode = False
                        staff_mode_active = False
                        waiting_for_response = False

                if waiting_for_response and (is_no(user_text) or is_natural_finish(user_text)):
                    print(f"🎯 Ответ '{user_text}' — естественное завершение просьб")
                    if all_requests:
                        requests_text = ", ".join(all_requests)
                    else:
                        requests_text = "הבקשה שלכם"
                    final_reply = f"העברתי את בקשתכם לצוות: {requests_text}. שיהיה לכם חופשה מהנה!"
                    speak(final_reply, language="he")
                    waiting_for_response = False
                    continue

                if waiting_for_response and is_yes(user_text):
                    print(f"🎯 Ответ '{user_text}' — ожидаем следующую просьбу")
                    continue_path = get_audio_path("continue_conversation")
                    if continue_path:
                        mic_paused = True
                        play_audio(continue_path)
                        mic_paused = False
                        print("🎤 Микрофон возобновлён")
                    else:
                        speak("מה עוד תרצו?", language="he")
                    update_activity()
                    continue

                if is_wifi_related(user_text):
                    audio_path = get_audio_path("info_wifi")
                    if audio_path:
                        print(f"📶 Интернет: {audio_path.name} ({VOICE_ENGINE})")
                        play_audio(audio_path)
                        update_activity()
                        continue

                print("⏳ AI определяет намерение...")
                result = ai_detect_intent(user_text, mode="main")
                intent = result.get("intent", "request")
                topic = result.get("topic")
                print(f"📋 Намерение: {intent}, Тема: {topic}")

                if is_urgent_request(user_text, intent):
                    if intent != "urgent":
                        print("🚨 Urgent: сработала защитная проверка критической фразы")
                    play_urgent_transfer(user_text)
                    continue

                if is_complaint_request(user_text, intent):
                    if intent != "complaint":
                        print("⚠️ Complaint: сработала защитная проверка явной жалобы")
                    play_complaint_transfer(user_text)
                    continue

                if intent == "ai_unavailable":
                    play_ai_unavailable_transfer()
                    continue

                if intent == "transfer":
                    print("👤 Перевод")
                    transfer_to_staff_mode = True
                    waiting_for_response = False
                    confirm_path = get_audio_path("transfer_confirm")
                    if confirm_path:
                        play_audio(confirm_path)
                        update_activity()
                    else:
                        speak("הבנתי. תרצו שאעביר אתכם לנציג אנושי?", language="he")
                    continue

                if intent == "separate":
                    print("📌 Смешанный запрос: прошу разделить вопрос и просьбу")
                    separate_path = get_audio_path("separate_request_or_question")
                    if separate_path:
                        # Запись, как и TTS, не должна попасть обратно в Whisper.
                        # Явно фиксируем окончание, чтобы realtime-тест мог подтвердить его.
                        mic_paused = True
                        played = play_audio(separate_path)
                        mic_paused = False
                        if played:
                            print(f"✅ Запись воспроизведена: {separate_path.name}")
                        print("🎤 Микрофон возобновлён")
                    else:
                        speak("אשמח אם תפנו אליי עם בקשה או שאלה בנפרד. לדוגמה: אני רוצה מגבות לחדר — זו בקשה, ואחר כך בנפרד מתי ארוחת הבוקר? — זו שאלה.")
                    update_activity()
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
                try:
                    response, not_understood, requests_list, wait_for_more, dialogue_action = ask_ai_with_memory(user_text, language="he")
                except AIProvidersUnavailable:
                    play_ai_unavailable_transfer()
                    continue

                if requests_list:
                    for req in requests_list:
                        if req and req not in all_requests:
                            all_requests.append(req)
                            print(f"📝 Добавлено в просьбы: '{req}'")

                # Не все естественные ответы удобно покрыть фиксированным
                # списком. DeepSeek уже получил историю разговора и может
                # безопасно определить, означает ли эта реплика «всё, спасибо».
                if waiting_for_response and dialogue_action == "finish" and not requests_list:
                    print(f"🎯 AI: '{user_text}' означает завершение просьб")
                    requests_text = ", ".join(all_requests) if all_requests else "הבקשה שלכם"
                    final_reply = f"העברתי את בקשתכם לצוות: {requests_text}. שיהיה לכם חופשה מהנה!"
                    waiting_for_response = False
                    speak(final_reply, language="he")
                    continue

                if waiting_for_response and dialogue_action == "continue" and not requests_list:
                    print(f"🎯 AI: '{user_text}' — гость продолжает разговор")
                    continue_path = get_audio_path("continue_conversation")
                    if continue_path:
                        mic_paused = True
                        play_audio(continue_path)
                        mic_paused = False
                        print("🎤 Микрофон возобновлён")
                    else:
                        speak("מה עוד תרצו?", language="he")
                    update_activity()
                    continue

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
                            finish_current_call()
                            continue
                else:
                    not_understood_count = 0
                    # Финальный вердикт имеет право создать только локальная
                    # ветка is_no() выше. Не даём модели завершить обычный
                    # звонок самовольно старой фразой из прежнего prompt.
                    if "שיהיה לכם חופשה מהנה" in response:
                        print("⚠️ AI попытался выдать финальный вердикт раньше ответа гостя")
                        response = "בשמחה, הבקשה נרשמה. האם אתם רוצים עוד משהו?"
                        wait_for_more = True
                    waiting_for_response = wait_for_more
                    speak(response, language="he")

            time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n👋 Завершение...")
            stop_music()
            stop_ding_loop()
            stop_recording_and_save()
            return False
        except Exception as e:
            print(f"❌ Ошибка main: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)


def run_continuous_service():
    """Последовательно обслуживает звонки, не перезагружая модели."""
    print("=" * 60)
    print("☎️ ПОСТОЯННЫЙ РЕЖИМ: модели остаются загруженными между звонками")
    print("Для остановки диспетчера нажмите Ctrl+C.")
    print("=" * 60)
    preload_static_audio()

    first_call = True
    while True:
        try:
            if first_call:
                configure_call(ROOM_NUMBER)
                first_call = False
            else:
                print("\n☎️ Готов к следующему звонку")
                next_room, _ = ask_room_number()
                configure_call(next_room)

            reset_call_runtime_state()
            result = main()
            stop_call_workers()
            if result is False:
                print("👋 Постоянный режим остановлен пользователем")
                return
            print("✅ Звонок завершён. Модели остаются в памяти.")
        except KeyboardInterrupt:
            stop_call_workers()
            print("\n👋 Постоянный режим остановлен")
            return


if __name__ == "__main__":
    if CONTINUOUS_SERVICE_MODE:
        run_continuous_service()
    else:
        main()
