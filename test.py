# test_faster_whisper.py
import os
import sys

# ⭐ Добавляем путь к CUDA DLL для CTranslate2
SITE_PACKAGES = r"C:\Users\david\AppData\Roaming\Python\Python312\site-packages"
ctranslate2_path = os.path.join(SITE_PACKAGES, "ctranslate2")
nvidia_cublas_path = os.path.join(SITE_PACKAGES, "nvidia", "cublas", "bin")
nvidia_runtime_path = os.path.join(SITE_PACKAGES, "nvidia", "cuda_runtime", "bin")

for path in [ctranslate2_path, nvidia_cublas_path, nvidia_runtime_path]:
    if os.path.exists(path):
        try:
            os.add_dll_directory(path)
            print(f"✅ DLL path: {path}")
        except Exception as e:
            print(f"⚠️ Не добавлен {path}: {e}")

# Отключаем предупреждение Hugging Face
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
import tempfile
import wave
import pyaudio
import numpy as np
from faster_whisper import WhisperModel

# ============================================================
# НАСТРОЙКИ
# ============================================================
MODEL_NAME = "ivrit-ai/whisper-large-v3-turbo-ct2"
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"   # ⭐ быстрее чем float16

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
DURATION = 5

# ============================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================
print("=" * 60)
print(f"🤖 Загрузка модели: {MODEL_NAME}")
print(f"   Device: {DEVICE} | Compute: {COMPUTE_TYPE}")
print("=" * 60)

try:
    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        num_workers=2,
    )
    print(f"✅ Модель загружена ({COMPUTE_TYPE})")
except Exception as e:
    print(f"⚠️ Ошибка с {COMPUTE_TYPE}: {e}")
    print("⚠️ Пробую float16...")
    try:
        COMPUTE_TYPE = "float16"
        model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
        print("✅ Модель загружена (float16)")
    except Exception as e2:
        print(f"⚠️ Ошибка с float16: {e2}")
        print("⚠️ Пробую CPU...")
        try:
            DEVICE = "cpu"
            COMPUTE_TYPE = "int8"
            model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
            print("✅ Модель загружена (CPU, int8)")
        except Exception as e3:
            print(f"❌ Всё упало: {e3}")
            sys.exit(1)

# ============================================================
# ЗАПИСЬ С МИКРОФОНА
# ============================================================
def record_audio(duration=5):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    print(f"\n🎤 Говорите ({duration} сек)...")
    frames = []
    
    total_chunks = int(RATE / CHUNK * duration)
    for i in range(total_chunks):
        elapsed = i * CHUNK / RATE
        bar = "█" * int(elapsed / duration * 30)
        print(f"\r[{bar:<30}] {elapsed:.1f}/{duration}с", end="", flush=True)
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    
    print()
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    
    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    return wav_path

# ============================================================
# РАСПОЗНАВАНИЕ (ОПТИМИЗИРОВАНО)
# ============================================================
def transcribe(wav_path):
    print("📤 Распознавание...")
    start = time.time()
    
    segments, info = model.transcribe(
        wav_path,
        language="he",
        beam_size=1,                        # ⭐ БЫЛО 5 — ГЛАВНОЕ УСКОРЕНИЕ
        best_of=1,                          # ⭐ не перебирать варианты
        temperature=0.0,
        condition_on_previous_text=False,   # ⭐ не тянуть контекст
        without_timestamps=True,            # ⭐ не считать тайминги
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        word_timestamps=False,
    )
    
    text = " ".join(segment.text for segment in segments).strip()
    elapsed = time.time() - start
    return text, elapsed

# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================
def main():
    print("\n" + "=" * 60)
    print(f"🧪 ТЕСТ РАСПОЗНАВАНИЯ ({DEVICE.upper()} / {COMPUTE_TYPE})")
    print("=" * 60)
    print("\nСкажите фразу на иврите (например: 'אפשר מגבות לחדר')")
    print("Или нажмите Ctrl+C для выхода.\n")
    
    test_count = 0
    total_time = 0
    
    while True:
        try:
            input("Нажмите ENTER чтобы записать 5 секунд... ")
            print()
            
            wav_path = record_audio(DURATION)
            text, elapsed = transcribe(wav_path)
            
            print("\n" + "=" * 60)
            print("📊 РЕЗУЛЬТАТ")
            print("=" * 60)
            print(f"📝 Распознано: {text}")
            print(f"⏱️  Время: {elapsed:.2f} сек")
            print("=" * 60)
            
            try:
                os.remove(wav_path)
            except:
                pass
            
            test_count += 1
            total_time += elapsed
            avg_time = total_time / test_count
            print(f"\n✅ Тест #{test_count} | Среднее время: {avg_time:.2f} сек")
            
        except KeyboardInterrupt:
            print("\n\n👋 Выход...")
            if test_count > 0:
                print(f"📊 Итого тестов: {test_count} | Среднее: {total_time/test_count:.2f} сек")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()