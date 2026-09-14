# voice_assistant_copy.py — КОПИЯ ОСНОВНОГО, НО С KOKORO
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
import io
import re
import urllib.request
from pathlib import Path
from datetime import datetime
import random
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

import pygame
import torch
import numpy as np
import soundfile as sf
import speech_recognition as sr
import edge_tts
from kokoro import KModel

# ============================================================
# ⭐ ПУТИ К KOKORO
# ============================================================

# ⭐ УСТАНАВЛИВАЕМ РАБОЧУЮ ПАПКУ

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
sys.path.insert(0, str(HERE))

KOKORO_BASE = Path("C:/Users/david/Desktop/kokoro-hebrew-main/inference")
KOKORO_MODEL_PATH = KOKORO_BASE / "kokoro_v1_hebrew.pth"
KOKORO_VOICE_PATH = KOKORO_BASE / "voices" / "he_shaul.pt"

# ============================================================
# ⭐ ПУТИ К АУДИО ФАЙЛАМ (KOKORO)
# ============================================================
AUDIO_DIR = Path(__file__).resolve().parent / "kokoro_audio"

# ============================================================
# ⭐ НАСТРОЙКИ
# ============================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

MUSIC_VOLUME = 0.13
GREETING_DELAY = 3.0

CONVERSATIONS_DIR = "conversations"
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

music_channel = pygame.mixer.Channel(0)
ding_channel = pygame.mixer.Channel(1)
voice_channel = pygame.mixer.Channel(2)
voice_channel.set_volume(1.0)   # ← МАКСИМАЛЬНАЯ ГРОМКОСТЬ ДЛЯ ГОЛОСА

# ============================================================
# ⭐ ЗАГРУЗКА KOKORO
# ============================================================
print("📥 Загрузка Kokoro...")
device = "cpu"
try:
    km = KModel(repo_id="hexgrad/Kokoro-82M", model=str(KOKORO_MODEL_PATH)).to(device).eval()
    voice = torch.load(KOKORO_VOICE_PATH, map_location=device, weights_only=True)
except Exception as e:
    print(f"❌ Ошибка загрузки Kokoro: {e}")
    sys.exit(1)

sys.path.insert(0, str(KOKORO_BASE))
try:
    from hebrew_g2p import phonemize_hebrew
except:
    print("⚠️ hebrew_g2p не найден, создаём заглушку")
    def phonemize_hebrew(text):
        return text, ""
print("✅ Kokoro загружен")

# ============================================================
# ⭐ ЗАГРУЗКА RE NIKUD
# ============================================================
try:
    from renikud_onnx import G2P
    renikud_model_path = HERE / "renikud_model.onnx"
    if not renikud_model_path.exists():
        print("📥 Скачивание ReNikud...")
        urllib.request.urlretrieve(
            "https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx",
            renikud_model_path
        )
    g2p = G2P(str(renikud_model_path))
    print("✅ ReNikud загружен")
except Exception as e:
    print(f"⚠️ ReNikud не загружен: {e}")
    g2p = None

# ============================================================
# ⭐ ФРАЗЫ ОЖИДАНИЯ
# ============================================================
WAITING_FILES = [
    str(AUDIO_DIR / "wait_ah_rag_kokoro.mp3"),
    str(AUDIO_DIR / "wait_bodeket_kokoro.mp3"),
    str(AUDIO_DIR / "wait_second_kokoro.mp3"),
    str(AUDIO_DIR / "wait_no_problem_kokoro.mp3"),
    str(AUDIO_DIR / "wait_just_second_kokoro.mp3"),
    str(AUDIO_DIR / "wait_give_me_kokoro.mp3"),
    str(AUDIO_DIR / "wait_daka_kokoro.mp3"),
    str(AUDIO_DIR / "wait_ok_second_kokoro.mp3"),
]

def play_random_waiting_phrase():
    try:
        file_name = random.choice(WAITING_FILES)
        file_path = Path(file_name)
        if not file_path.exists():
            print(f"⚠️ Файл {file_name} не найден.")
            play_ding()
            return
        print(f"🎵 Ожидание: {file_name}")
        sound = pygame.mixer.Sound(file_path)
        sound.set_volume(0.4)
        ding_channel.play(sound)
    except Exception as e:
        print(f"⚠️ Ошибка: {e}")
        play_ding()

# ============================================================
# ⭐ КЛЮЧЕВЫЕ СЛОВА (ОБНОВЛЕННЫЕ - РАЗДЕЛЕННЫЕ ТРАПЕЗЫ)
# ============================================================
INFO_KEYWORDS = {
    "lobby": ["לובי", "לובי המלון", "איפה הלובי", "שעות לובי"],
    "synagogue": ["בית כנסת", "בית הכנסת", "תפילה", "שעות תפילה", "בית כנסת שעות"],
    "spa": ["ספא", "SPA", "royal spa", "בריכת מלח", "סאונה", "ג'קוזי", "חדר כושר", "איפה הספא", "איפה ספא", "קומת ספא", "באיזו קומה"],
    "spa_transfer": ["עיסוי", "מסאז'", "טיפול", "מחיר", "עלות", "כמה עולה", "רפלקסולוגיה", "פילינג", "עטיפה", "בוץ", "אלוורה", "מחירון"],
    "pool": ["בריכה", "בריכת", "בריכה חיצונית", "שעות בריכה", "מים מתוקים"],
    "beach": ["חוף", "חוף הים", "ים המלח", "יציאה לחוף"],
    # ⭐ ТРАПЕЗЫ РАЗДЕЛЕНЫ
    "breakfast": ["ארוחת בוקר", "בוקר", "ארוחה בוקר", "שעות ארוחת בוקר"],
    "lunch": ["ארוחת צהריים", "צהריים", "ארוחת צהרים", "שעות ארוחת צהריים"],
    "dinner": ["ארוחת ערב", "ערב", "ארוחת ערב", "שעות ארוחת ערב"],
    "all_meals": ["ארוחה", "אוכל", "מזנון", "שעות ארוחה", "ארוחות", "ארוחת"],
    "hours": ["שעות", "מה השעות", "מתי פתוח", "מתי סגור", "זמני", "רוחות"],
    "rules": ["עזיבת חדרים", "עישון", "צוות הקבלה", "יש לעזוב", "11:00"],
    # ⭐ НОВЫЕ КЛЮЧЕВЫЕ СЛОВА ДЛЯ ПЕРЕВОДА НА ПЕРСОНАЛА
    "transfer_to_staff": ["תעביר אותי לצוות", "תעביר לנציג", "דבר עם נציג", "אני רוצה לדבר עם נציג", "תעביר אותי לנציג", "שלח אותי לצוות", "אני רוצה לדבר עם איש צוות"]
}

INFO_FILES = {
    "lobby": AUDIO_DIR / "info_lobby_kokoro.mp3",
    "synagogue": AUDIO_DIR / "info_synagogue_kokoro.mp3",
    "spa": AUDIO_DIR / "info_spa_kokoro.mp3",
    "spa_transfer": AUDIO_DIR / "info_spa_transfer_kokoro.mp3",
    "pool": AUDIO_DIR / "info_pool_kokoro.mp3",
    "beach": AUDIO_DIR / "info_beach_kokoro.mp3",
    # ⭐ ОТДЕЛЬНЫЕ ФАЙЛЫ ДЛЯ КАЖДОЙ ТРАПЕЗЫ
    "breakfast": AUDIO_DIR / "info_breakfast_kokoro.mp3",
    "lunch": AUDIO_DIR / "info_lunch_kokoro.mp3",
    "dinner": AUDIO_DIR / "info_dinner_kokoro.mp3",
    "all_meals": AUDIO_DIR / "info_restaurant_kokoro.mp3",
    "hours": AUDIO_DIR / "info_hours_kokoro.mp3",
    "rules": AUDIO_DIR / "info_rules_kokoro.mp3",
    # ⭐ НОВЫЕ ФАЙЛЫ ДЛЯ ПЕРЕВОДА НА ПЕРСОНАЛА
    "transfer_to_staff": AUDIO_DIR / "transfer_confirm_kokoro.mp3"
}

# ============================================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ============================================================
audio_queue = queue.Queue()
is_speaking = False
microphone_active = False
mic_thread = None
stop_mic = False
conversation_history = []
MAX_HISTORY = 20
stop_loading_ding = False

recording_frames = []
is_recording = False
recording_lock = threading.Lock()
current_conversation_text = []
recording_thread = None
stop_recording_flag = False

info_played = False
last_info_file = None
# ⭐ ПЕРЕМЕННАЯ ДЛЯ ОТСЛЕЖИВАНИЯ РЕЖИМА СПА-ТРАНСФЕРА
spa_transfer_mode = False
# ⭐ НОВАЯ ПЕРЕМЕННАЯ ДЛЯ ОТСЛЕЖИВАНИЯ РЕЖИМА ПЕРЕВОДА НА ПЕРСОНАЛА
transfer_to_staff_mode = False

# ============================================================
# ⭐ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================
def extract_requests_from_verdict(text):
    pattern = r'העברתי את בקשתכם לצוות:\s*(.+?)\s*\.\s*שיהיה לכם חופשה מהנה'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return ""

# ============================================================
# ⭐ ОТПРАВКА В WHATSAPP (ЧЕРЕЗ ОЧЕРЕДЬ)
# ============================================================
QUEUE_FILE = Path(r"C:\Users\david\Desktop\WhatsApp\Project\send_queue.json")

def send_verdict_to_whatsapp(requests_text, audio_file_path=None):
    """Сохраняет сообщение и аудио в очередь для отправки через reaction_monitor.py"""
    
    print("🔍 send_verdict_to_whatsapp ВЫЗВАНА!")
    
    if not requests_text:
        requests_text = "בקשה לא זוהתה"
    
    message = f"""🎤 *הודעה מהמערכת הדיגיטלית* (AI)

📋 *האורח ביקש:*
{requests_text}"""

    print(f"📝 Сообщение: {message[:50]}...")

    queue = []
    if QUEUE_FILE.exists():
        try:
            with open(QUEUE_FILE, "r", encoding="utf-8") as f:
                queue = json.load(f)
            print(f"📂 Загружено {len(queue)} сообщений из очереди")
        except Exception as e:
            print(f"⚠️ Ошибка чтения очереди: {e}")
            queue = []
    
    queue.append({
        "id": str(int(time.time() * 1000)),
        "text": message,
        "audio_file": audio_file_path,
        "created_at": time.time()
    })
    
    try:
        with open(QUEUE_FILE, "w", encoding="utf-8") as f:
            json.dump(queue, f, ensure_ascii=False, indent=4)
        print(f"📤 Сообщение добавлено в очередь отправки (всего: {len(queue)})")
        return True
    except Exception as e:
        print(f"❌ Ошибка записи в очередь: {e}")
        return False

# ============================================================
# ⭐ УМНОЕ ОПРЕДЕЛЕНИЕ ЗАПРОСА О ЕДЕ
# ============================================================
def check_food_request(text):
    """
    Определяет, о какой именно еде спрашивает гость
    Возвращает ключ для INFO_FILES
    """
    text_lower = text.lower()
    
    # Сначала проверяем конкретные запросы
    if "ארוחת בוקר" in text_lower or "ארוחה בוקר" in text_lower:
        return "breakfast"
    elif "ארוחת צהריים" in text_lower or "ארוחת צהרים" in text_lower:
        return "lunch"
    elif "ארוחת ערב" in text_lower:
        return "dinner"
    # Если запрос общий - про все трапезы
    elif "ארוחה" in text_lower or "אוכל" in text_lower or "מזנון" in text_lower or "ארוחות" in text_lower:
        return "all_meals"
    
    return None

# ============================================================
# ⭐ ИНФОРМАЦИОННЫЕ ЗАПРОСЫ (ОБНОВЛЕННАЯ)
# ============================================================
def check_info_request(text):
    """Проверяет, является ли текст информационным запросом"""
    text_lower = text.lower()
    
    # ⭐ СНАЧАЛА ПРОВЕРЯЕМ ЗАПРОСЫ О ЕДЕ
    food_category = check_food_request(text)
    if food_category:
        file_path = INFO_FILES.get(food_category)
        if file_path and file_path.exists():
            print(f"🔍 Распознан запрос о еде: {food_category}")
            return file_path
    
    # ПОТОМ ВСЕ ОСТАЛЬНЫЕ КАТЕГОРИИ
    for category, keywords in INFO_KEYWORDS.items():
        # Пропускаем категории еды, так как мы их уже обработали
        if category in ["breakfast", "lunch", "dinner", "all_meals"]:
            continue
        for keyword in keywords:
            if keyword in text_lower:
                print(f"🔍 Распознан информационный запрос: {category}")
                return INFO_FILES[category]
    return None

def play_info_file(file_path):
    global info_played, last_info_file, spa_transfer_mode, transfer_to_staff_mode
    try:
        if not Path(file_path).exists():
            print(f"❌ Файл {file_path} не найден.")
            return False
        info_played = True
        last_info_file = file_path
        
        # ⭐ ЕСЛИ ЭТО ФАЙЛ СПА-ТРАНСФЕРА - ВКЛЮЧАЕМ РЕЖИМ
        if "info_spa_transfer" in str(file_path):
            spa_transfer_mode = True
            print("🧖 ВКЛЮЧЕН РЕЖИМ СПА-ТРАНСФЕРА")
        
        # ⭐ ЕСЛИ ЭТО ФАЙЛ ПЕРЕВОДА НА ПЕРСОНАЛА - ВКЛЮЧАЕМ РЕЖИМ
        if "transfer_confirm" in str(file_path):
            transfer_to_staff_mode = True
            print("👤 ВКЛЮЧЕН РЕЖИМ ПЕРЕВОДА НА ПЕРСОНАЛА")
        
        print(f"🎤 Воспроизведение информации: {file_path}")
        stop_microphone()
        time.sleep(0.2)
        sound = pygame.mixer.Sound(file_path)
        voice_channel.play(sound)
        while voice_channel.get_busy():
            time.sleep(0.05)
        print("✅ Информация воспроизведена")
        start_microphone()
        return True
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        start_microphone()
        return False

# ============================================================
# ⭐ ЗАПИСЬ РАЗГОВОРА (ИСПРАВЛЕННАЯ — БЕЗ ЗАВИСАНИЙ)
# ============================================================
def recording_worker():
    """Постоянно записывает аудио с микрофона в отдельном потоке"""
    global recording_frames, is_recording, stop_recording_flag
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    p = pyaudio.PyAudio()
    stream = None
    
    while not stop_recording_flag:
        try:
            if stream is None:
                stream = p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK
                )
                with recording_lock:
                    is_recording = True
            
            # ⭐ ПРОВЕРЯЕМ, НЕ НУЖНО ЛИ ОСТАНОВИТЬСЯ
            if stop_recording_flag:
                break
            
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            # ⭐ ЕЩЁ РАЗ ПРОВЕРЯЕМ
            if stop_recording_flag:
                break
            
            with recording_lock:
                if is_recording:
                    recording_frames.append(data)
                    
        except Exception as e:
            print(f"⚠️ Ошибка записи: {e}")
            time.sleep(0.1)
    
    if stream:
        stream.stop_stream()
        stream.close()
    p.terminate()
    with recording_lock:
        is_recording = False
    print("🎙️ Запись остановлена")
    
def start_recording():
    """Запускает постоянную запись в отдельном потоке"""
    global recording_thread, stop_recording_flag, recording_frames, current_conversation_text
    
    with recording_lock:
        recording_frames = []
        current_conversation_text = []
        stop_recording_flag = False
    
    # ⭐ ПРОВЕРЯЕМ, ЧТО ПОТОК НЕ ЗАВИС
    if recording_thread and recording_thread.is_alive():
        print("⚠️ Поток записи уже запущен, перезапускаем...")
        stop_recording_flag = True
        recording_thread.join(timeout=1.0)
    
    recording_thread = threading.Thread(target=recording_worker, daemon=True)
    recording_thread.start()
    print("🎙️ Запись запущена (постоянно)")



def stop_recording_and_save():
    """Останавливает запись и сохраняет файл (БЕЗ ЗАВИСАНИЙ)"""
    global stop_recording_flag, recording_frames, current_conversation_text
    
    print("⏹️ Остановка записи...")
    
    # ⭐ 1. ОСТАНАВЛИВАЕМ ПОТОК ЗАПИСИ (без блокировки)
    stop_recording_flag = True
    
    # ⭐ 2. ЖДЁМ ЗАВЕРШЕНИЯ ПОТОКА (НО НЕ БОЛЬШЕ 1 СЕКУНДЫ)
    if recording_thread and recording_thread.is_alive():
        print("⏳ Ожидание завершения потока записи...")
        recording_thread.join(timeout=1.0)
        if recording_thread.is_alive():
            print("⚠️ Поток записи не завершился, принудительно продолжаем...")
    
    # ⭐ 3. КОПИРУЕМ ДАННЫЕ БЕЗ БЛОКИРОВКИ (или с короткой блокировкой)
    frames_copy = []
    text_copy = []
    
    try:
        with recording_lock:
            frames_copy = recording_frames.copy()
            text_copy = current_conversation_text.copy()
            # ⭐ ОЧИЩАЕМ СРАЗУ ПОСЛЕ КОПИРОВАНИЯ
            recording_frames = []
            current_conversation_text = []
    except Exception as e:
        print(f"⚠️ Ошибка копирования данных: {e}")
        return None
    
    # ⭐ 4. ПРОВЕРЯЕМ ДАННЫЕ
    if not frames_copy:
        print("⚠️ Нет данных для сохранения")
        return None
    
    print(f"📊 Сохранение {len(frames_copy)} фреймов...")
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"conversation_{timestamp}"
        mp3_path = os.path.abspath(os.path.join(CONVERSATIONS_DIR, f"{base_filename}.mp3"))
        txt_path = os.path.abspath(os.path.join(CONVERSATIONS_DIR, f"{base_filename}.txt"))
        
        # ⭐ СОХРАНЯЕМ WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b''.join(frames_copy))
        
        # ⭐ КОНВЕРТИРУЕМ В MP3
        try:
            import subprocess
            subprocess.run([
                "ffmpeg", "-y", "-i", wav_path,
                "-acodec", "libmp3lame", "-ab", "64k",
                mp3_path
            ], capture_output=True, check=True, timeout=5)
            print(f"✅ Запись сохранена: {mp3_path}")
            os.remove(wav_path)
        except Exception as e:
            print(f"⚠️ Ошибка конвертации: {e}")
            wav_final = os.path.join(CONVERSATIONS_DIR, f"{base_filename}.wav")
            os.rename(wav_path, wav_final)
            print(f"✅ Запись сохранена как WAV: {wav_final}")
            mp3_path = wav_final
        
        # ⭐ СОХРАНЯЕМ ТЕКСТ
        if text_copy:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Диалог {timestamp}\n")
                f.write("=" * 50 + "\n")
                for line in text_copy:
                    f.write(line + "\n")
            print(f"✅ Текст диалога сохранен: {txt_path}")
        
        return mp3_path
        
    except Exception as e:
        print(f"❌ Ошибка сохранения записи: {e}")
        return None


def add_conversation_text(speaker, text):
    with recording_lock:
        timestamp = datetime.now().strftime("%H:%M:%S")
        current_conversation_text.append(f"[{timestamp}] {speaker}: {text}")

# ============================================================
# 🎵 МУЗЫКА И ДИНЬ
# ============================================================
def start_music():
    try:
        music_channel.play(pygame.mixer.Sound("background.mp3"), loops=-1)
        music_channel.set_volume(MUSIC_VOLUME)
        print(f"🎵 מוזיקה מופעלת ({int(MUSIC_VOLUME * 100)}%)")
    except:
        pass

def stop_music():
    music_channel.stop()

def play_ding():
    try:
        sound = pygame.mixer.Sound("ding.mp3")
        sound.set_volume(0.35)
        ding_channel.play(sound)
    except:
        pass

def start_loading_wait():
    global stop_loading_ding
    stop_loading_ding = False
    def _play_once():
        time.sleep(0.3)
        if not stop_loading_ding:
            play_random_waiting_phrase()
    thread = threading.Thread(target=_play_once, daemon=True)
    thread.start()

def stop_loading_ding_signal():
    global stop_loading_ding
    stop_loading_ding = True

# ============================================================
# 🎤 МИКРОФОН
# ============================================================
def mic_worker():
    global microphone_active, stop_mic
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    p = pyaudio.PyAudio()
    stream = None
    last_speech_end = 0
    while not stop_mic:
        try:
            if stream is None:
                stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
                microphone_active = True
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_array = np.frombuffer(data, dtype=np.int16)
            if len(audio_array) == 0:
                continue
            mean_square = np.mean(audio_array.astype(np.float32) ** 2)
            if mean_square <= 0:
                continue
            energy = np.sqrt(mean_square)
            if energy > 500 and (time.time() - last_speech_end) > 1.5:
                frames = [data]
                silence_count = 0
                for _ in range(60):
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    if len(audio_array) == 0:
                        continue
                    mean_square = np.mean(audio_array.astype(np.float32) ** 2)
                    if mean_square > 0:
                        energy = np.sqrt(mean_square)
                    else:
                        energy = 0
                    if energy > 500:
                        frames.append(data)
                        silence_count = 0
                    else:
                        silence_count += 1
                        frames.append(data)
                        if silence_count > 12:
                            break
                if stream:
                    stream.stop_stream()
                    stream.close()
                    stream = None
                    microphone_active = False
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))
                wav_buffer.seek(0)
                recognizer = sr.Recognizer()
                with sr.AudioFile(wav_buffer) as source:
                    audio = recognizer.record(source)
                    try:
                        text = recognizer.recognize_google(audio, language="he-IL")
                        if text and len(text) > 2:
                            print(f"🎤 זוהתה דיבור: {text}")
                            audio_queue.put(text)
                            add_conversation_text("Гость", text)
                    except:
                        pass
                last_speech_end = time.time()
                if not stop_mic:
                    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
                    microphone_active = True
        except Exception as e:
            print(f"⚠️ שגיאת מיקרופון: {e}")
            time.sleep(0.1)
    if stream:
        stream.stop_stream()
        stream.close()
    p.terminate()
    microphone_active = False
    print("🎤 מיקרופון כבוי")

def start_microphone():
    global mic_thread, stop_mic
    stop_mic = False
    mic_thread = threading.Thread(target=mic_worker, daemon=True)
    mic_thread.start()

def stop_microphone():
    global stop_mic, microphone_active
    stop_mic = True
    microphone_active = False
    time.sleep(0.2)

# ============================================================
# 🗣️ ГОВОРИТЬ (KOKORO)
# ============================================================
def speak(text, language="he"):
    global is_speaking
    print(f"🤖 AI: {text}")
    is_speaking = True
    add_conversation_text("ИИ", text)
    stop_microphone()
    stop_loading_ding_signal()
    time.sleep(0.2)
    
    try:
        # 1. Огласовки
        if g2p:
            try:
                text_with_niqqud = g2p.phonemize(text)
            except:
                text_with_niqqud = text
        else:
            text_with_niqqud = text
        
        # 2. Генерация аудио
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
            audio_data = np.concatenate(pieces)
            
            # ⭐ УСИЛИВАЕМ ГРОМКОСТЬ
            audio_data = audio_data * 5.0
            
            # ⭐ ОГРАНИЧИВАЕМ, ЧТОБЫ НЕ БЫЛО КЛИППИНГА
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
        else:
            speak_fallback(text)
            
    except Exception as e:
        print(f"❌ Ошибка Kokoro: {e}")
        speak_fallback(text)
    
    is_speaking = False
    time.sleep(0.5)
    
    # ⭐ ПРОВЕРКА ФИНАЛЬНОГО ВЕРДИКТА
    if "שיהיה לכם חופשה מהנה" in text:
        print(f"✅ ФИНАЛЬНЫЙ ВЕРДИКТ: {text}")
        requests_text = extract_requests_from_verdict(text)
        print(f"📋 Просьбы: {requests_text}")
        
        audio_path = None
        try:
            audio_path = stop_recording_and_save()
        except Exception as e:
            print(f"⚠️ Ошибка сохранения записи: {e}")
            audio_path = None
        
        send_verdict_to_whatsapp(requests_text, audio_path)
        
        print("👋 Завершение работы...")
        stop_music()
        stop_microphone()
        stop_loading_ding_signal()
        os._exit(0)

    print("🎤 Включение микрофона после ответа...")
    start_microphone()
    
def speak_fallback(text, language="he"):
    try:
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
    except Exception as e:
        print(f"❌ Ошибка fallback: {e}")
        
# ============================================================
# 🔤 AI С ПАМЯТЬЮ
# ============================================================
def ask_ai_with_memory(prompt: str, language: str = "he") -> tuple:
    global conversation_history
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_URL,
            timeout=15.0
        )
        
        system_prompt = """
אתה העוזר הדיגיטלי של מלון רויאל ים המלח.

📌 הכללים החשובים ביותר:

1. אתה תמיד זוכר מה האורח ביקש קודם.
2. אם האורח מוסיף בקשות חדשות - אתה מוסיף אותן לרשימה.
3. אתה יכול לשאול "האם אתם רוצים עוד משהו?" רק בפעם הראשונה!
4. אם האורח אומר "כן" או "לא תודה" - אתה נותן פסק דין סופי.
5. אתה יכול לעשות 2 ניסיונות הבהרה לכל היותר.
6. אם לא הבנת אחרי 2 ניסיונות - אתה אומר "מצטער. לא הצלחתי להבין." ומפסיק.

📝 תבניות תשובה:

🔹 תשובה ראשונה (בקשה רגילה):
"בשמחה נשלח לכם [רשימת בקשות]. האם אתם רוצים עוד משהו?"

🔹 תשובה ראשונה (רק אם האורח באמת כועס):
"אני מבין את התסכול. אשמח לטפל: [רשימת בקשות]. העברתי את בקשתכם לצוות. האם אתם רוצים עוד משהו?"

🔹 תשובה שנייה (אם האורח מוסיף בקשה חדשה):
"הבנתי. אז אתם רוצים [רשימת בקשות 1] ו-[רשימת בקשות 2], נכון?"

🔹 תשובה שלישית (אם האורח ענה "נכון" או "כן" או "לא תודה"):
"העברתי את בקשתכם לצוות: [רשימת כל הבקשות]. שיהיה לכם חופשה מהנה!"

🔹 אם לא הבנת בפעם הראשונה:
"סלחו לי. כדי שאוכל לעזור לכם טוב יותר, אשמח להבין בדיוק מה אתם צריכים."

🔹 אם לא הבנת אחרי ההבהרה:
"מצטער. לא הצלחתי להבין את בקשתכם." - זו המילה האחרונה שלך!

🚫 אסור לומר:
- "מצטער" ו"אני מבין את התסכול" - רק אם האורח באמת כועס או מתלונן!
- יותר מ-2 משפטים בתשובה!
- "העברתי את בקשתכם לצוות" - רק בפסק דין סופי! (תשובה שלישית)

✅ חובה:
- לדבר תמיד בלשון רבים (לכם / אתם) - זה מנומס
- לקצר! תשובה מקסימום 60-70 תווים
- לאשר את הבקשות בדיוק כפי שהאורח אמר
- לשמור היסטוריית בקשות!
- אם האורח מוסיף בקשות - לעדכן את הרשימה!
"""
        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history[-10:]:
            messages.append(msg)
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=80,
            stream=False
        )
        answer = response.choices[0].message.content.strip()
        conversation_history.append({"role": "user", "content": prompt})
        conversation_history.append({"role": "assistant", "content": answer})
        if len(conversation_history) > MAX_HISTORY:
            conversation_history = conversation_history[-MAX_HISTORY:]
        not_understood = "לא הצלחתי להבין" in answer or "לא הבנתי" in answer
        return answer, not_understood
    except Exception as e:
        return f"❌ שגיאה: {e}", False

# ============================================================
# ⭐ SPA_TRANSFER (ИСПРАВЛЕННАЯ ЛОГИКА)
# ============================================================
def check_spa_transfer_response(text):
    """Проверяет ответ гостя на вопрос о переводе в спа-отдел"""
    global spa_transfer_mode
    if not spa_transfer_mode:
        return None
    
    text_lower = text.lower()
    
    # ⭐ СНАЧАЛА ПРОВЕРЯЕМ ОТРИЦАТЕЛЬНЫЕ ОТВЕТЫ (С "לא")
    negative_patterns = [
        "לא תודה", "לא, תודה", "לא רוצה", "לא צריך", "לא מעוניין", 
        "לא מעוניינת", "לא", "לא תודה", "לא,", "לא רוצה", "לא צריך"
    ]
    
    for pattern in negative_patterns:
        if pattern in text_lower:
            print("✅ Гость НЕ хочет говорить со спа-отделом → завершаем диалог")
            return "end"
    
    # ПОТОМ ПРОВЕРЯЕМ ПОЛОЖИТЕЛЬНЫЕ ОТВЕТЫ
    positive = ["כן", "כן בבקשה", "בבקשה", "אני רוצה", "אשמח", "יאללה", "סבבה", "אוקי", "טוב", 
                "מעוניין", "מעוניינת", "רוצה", "אני מעוניין", "אני מעוניינת", "כן אני"]
    
    for word in positive:
        if word in text_lower:
            print("🔁 Гость хочет поговорить со спа-отделом → перевод на персонала")
            return "transfer"
    
    # ⭐ ЕСЛИ НИЧЕГО НЕ РАСПОЗНАНО - ЭТО ДРУГОЙ ЗАПРОС
    print("🔄 Гость задал другой вопрос. Выходим из режима спа-трансфера.")
    return "other"

# ============================================================
# ⭐ НОВАЯ ЛОГИКА ДЛЯ ПЕРЕВОДА НА ПЕРСОНАЛА
# ============================================================
def check_transfer_to_staff_response(text):
    """Проверяет ответ гостя на вопрос о переводе на персонала"""
    global transfer_to_staff_mode
    if not transfer_to_staff_mode:
        return None
    
    text_lower = text.lower()
    
    # ⭐ СНАЧАЛА ПРОВЕРЯЕМ ОТРИЦАТЕЛЬНЫЕ ОТВЕТЫ
    negative_patterns = [
        "לא", "לא תודה", "לא, תודה", "לא רוצה", "לא צריך", "לא מעוניין", 
        "לא מעוניינת", "לא,", "לא רוצה", "לא צריך"
    ]
    
    for pattern in negative_patterns:
        if pattern in text_lower:
            print("✅ Гость НЕ хочет перевода на персонала → продолжаем диалог")
            return "end"
    
    # ПОТОМ ПРОВЕРЯЕМ ПОЛОЖИТЕЛЬНЫЕ ОТВЕТЫ
    positive = ["כן", "כן בבקשה", "בבקשה", "אני רוצה", "אשמח", "יאללה", "סבבה", "אוקי", "טוב", 
                "מעוניין", "מעוניינת", "רוצה", "אני מעוניין", "אני מעוניינת", "כן אני"]
    
    for word in positive:
        if word in text_lower:
            print("🔁 Гость хочет перевода на персонала → выполняем перевод")
            return "transfer"
    
    # Если ничего не распознано
    return None

def play_transfer_complete():
    """Завершает диалог с переводом на персонала"""
    print('=' * 60)
    print('👤 ЗАВЕРШИТЬ ЛОГИКУ КОГДА ПРОГРАММА БУДЕТ ГОТОВА')
    print('=' * 60)
    
    try:
        # Воспроизводим финальную фразу
        complete_file = AUDIO_DIR / "transfer_complete_kokoro.mp3"
        if complete_file.exists():
            print(f"🎤 Воспроизведение: {complete_file}")
            stop_microphone()
            time.sleep(0.2)
            sound = pygame.mixer.Sound(complete_file)
            voice_channel.play(sound)
            while voice_channel.get_busy():
                time.sleep(0.05)
        
        # Останавливаем запись
        try:
            stop_recording_and_save()
        except Exception as e:
            print(f"⚠️ Ошибка сохранения записи: {e}")
        
        print("👋 Завершение работы...")
        stop_music()
        stop_microphone()
        stop_loading_ding_signal()
        
        time.sleep(1)
        os._exit(0)
        
    except Exception as e:
        print(f"❌ Ошибка при завершении: {e}")
        os._exit(1)

def play_spa_transfer_complete():
    """Завершает диалог со спа-трансфером"""
    print('=' * 60)
    print('🧖 ДОДЕЛАТЬ ЛОГИКУ ПЕРЕВОДА В СПА, КОГДА БУДЕМ УСТАНАВЛИВАТЬ ПРОГРАММУ!!!')
    print('=' * 60)
    
    try:
        # Останавливаем запись
        try:
            stop_recording_and_save()
        except Exception as e:
            print(f"⚠️ Ошибка сохранения записи: {e}")
        
        print("👋 Завершение работы...")
        stop_music()
        stop_microphone()
        stop_loading_ding_signal()
        
        # Ждем немного перед завершением
        time.sleep(1)
        os._exit(0)
        
    except Exception as e:
        print(f"❌ Ошибка при завершении: {e}")
        os._exit(1)

def play_error_transfer():
    try:
        error_file = AUDIO_DIR / "error_kokoro.mp3"
        if not error_file.exists():
            speak("מצטער. לא הצלחתי להבין את בקשתכם. מעביר אתכם עכשיו לצוות.", language="he")
            return False
        stop_microphone()
        stop_loading_ding_signal()
        time.sleep(0.2)
        sound = pygame.mixer.Sound(error_file)
        voice_channel.play(sound)
        while voice_channel.get_busy():
            time.sleep(0.05)
        print("✅ Перевод на персонала завершен")
        time.sleep(0.5)
        # ⭐ СОХРАНЯЕМ ЗАПИСЬ С ЗАЩИТОЙ
        try:
            stop_recording_and_save()
        except:
            pass
        print("👋 Завершение работы...")
        stop_music()
        stop_microphone()
        stop_loading_ding_signal()
        os._exit(0)
        return True
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        start_microphone()
        return False

def play_greeting_with_delay():
    try:
        greeting_file = AUDIO_DIR / "greeting_kokoro.mp3"
        if not greeting_file.exists():
            print(f"❌ Файл {greeting_file} не найден.")
            return False
        print("🎤 Остановка микрофона...")
        stop_microphone()
        time.sleep(0.3)
        print(f"⏳ Ожидание {GREETING_DELAY} секунд перед приветствием...")
        for i in range(int(GREETING_DELAY)):
            print(f"   {GREETING_DELAY - i}...")
            time.sleep(1)
        print("🎤 Воспроизведение приветствия...")
        sound = pygame.mixer.Sound(greeting_file)
        voice_channel.play(sound)
        while voice_channel.get_busy():
            time.sleep(0.05)
        print("✅ Приветствие завершено")
        time.sleep(0.5)
        print("🎤 Включение микрофона...")
        start_microphone()
        return True
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        start_microphone()
        return False

# ============================================================
# 🚀 MAIN
# ============================================================
def main():
    global info_played, last_info_file, spa_transfer_mode, transfer_to_staff_mode
    print("=" * 60)
    print("🎤 עוזר קולי למלון (KOKORO) — РЕЗЕРВ")
    print("=" * 60)
    print("  דברו במיקרופון")
    print("  אמור 'יציאה' כדי לסיים")
    print("=" * 60)
    
    start_recording()
    start_music()
    time.sleep(0.5)
    print(f"\n⏳ Ожидание {GREETING_DELAY} секунд перед приветствием...")
    play_greeting_with_delay()
    print("\n🎤 Микрофон активен. Говорите...")
    
    while True:
        try:
            if not audio_queue.empty():
                user_text = audio_queue.get()
                if user_text.lower() in ["יציאה", "להתראות", "exit", "quit"]:
                    speak("להתראות! שיהיה לך יום טוב.", language="he")
                    stop_music()
                    stop_microphone()
                    stop_recording_and_save()
                    break
                
                # ⭐ ОБРАБОТКА ПЕРЕВОДА НА ПЕРСОНАЛА (ПРОВЕРЯЕМ ПЕРВЫМ!)
                if transfer_to_staff_mode:
                    transfer_response = check_transfer_to_staff_response(user_text)
                    if transfer_response == "transfer":
                        print("🔁 Перевод на персонала...")
                        play_transfer_complete()
                        continue
                    elif transfer_response == "end":
                        print("👤 Гость отказался от перевода. Продолжаем диалог.")
                        transfer_to_staff_mode = False
                        # Продолжаем обработку запроса
                    else:
                        # Ответ не распознан - ждем дальше
                        print("🤔 Ответ не распознан. Ждем 'כן' или 'לא'...")
                        continue
                
                # ⭐ ОБРАБОТКА СПА-ТРАНСФЕРА
                if spa_transfer_mode:
                    spa_response = check_spa_transfer_response(user_text)
                    if spa_response == "transfer":
                        print("🔁 Перевод на спа-отдел...")
                        play_spa_transfer_complete()
                        continue
                    elif spa_response == "end":
                        print("👋 Гость отказался от перевода. Завершаем диалог.")
                        try:
                            stop_recording_and_save()
                        except:
                            pass
                        print("👋 Завершение работы...")
                        stop_music()
                        stop_microphone()
                        stop_loading_ding_signal()
                        os._exit(0)
                    elif spa_response == "other":
                        # ⭐ ВЫХОДИМ ИЗ РЕЖИМА СПА-ТРАНСФЕРА
                        spa_transfer_mode = False
                        print("🔄 Обрабатываем запрос как обычный...")
                    else:
                        # Ответ не распознан - ждем дальше
                        print("🤔 Ответ не распознан. Ждем 'כן' или 'לא'...")
                        continue
                
                info_file = check_info_request(user_text)
                if info_file:
                    print(f"ℹ️ Информационный запрос: воспроизведение {info_file}")
                    play_info_file(info_file)
                    add_conversation_text("ИИ", f"[АУДИО-ИНФОРМАЦИЯ] {info_file}")
                    continue
                
                print("⏳ חושב...")
                start_loading_wait()
                response, not_understood = ask_ai_with_memory(user_text, language="he")
                stop_loading_ding_signal()
                
                if response.startswith("❌"):
                    print(response)
                    speak("סליחה, הייתה שגיאה. נסה שוב.", language="he")
                    continue
                
                if not_understood:
                    print("❌ ИИ не понял запрос. Воспроизведение error_elevenlabs.mp3")
                    play_error_transfer()
                    conversation_history = []
                    info_played = False
                    last_info_file = None
                    spa_transfer_mode = False
                    transfer_to_staff_mode = False
                else:
                    speak(response, language="he")
                    
            time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n👋 סיום...")
            stop_music()
            stop_microphone()
            stop_loading_ding_signal()
            stop_recording_and_save()
            break
        except Exception as e:
            print(f"❌ שגיאה: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
