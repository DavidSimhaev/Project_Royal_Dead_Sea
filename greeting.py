# greeting_renikud_fixed.py
import sys
import re
from pathlib import Path
import torch
import numpy as np
import soundfile as sf
from kokoro import KModel
import urllib.request
import os

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODEL_PATH = HERE / "kokoro_v1_hebrew.pth"
VOICE_PATH = HERE / "voices" / "he_shaul.pt"

print("=" * 70)
print("🔊 СИНТЕЗ РЕЧИ НА ИВРИТЕ (ReNikud - автоматические огласовки)")
print("=" * 70)

# Загрузка ReNikud
try:
    from renikud_onnx import G2P
    
    # Путь к модели
    model_path = HERE / "renikud_model.onnx"
    
    # Скачиваем модель если нет
    if not model_path.exists():
        print("📥 Скачивание модели ReNikud...")
        # Правильная ссылка из официального репозитория
        url = "https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx"
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"✅ Модель скачана ({model_path.stat().st_size / 1024 / 1024:.2f} MB)")
        except Exception as e:
            print(f"❌ Ошибка скачивания: {e}")
            # Если не работает, пробуем альтернативный вариант
            print("Пробуем альтернативный URL...")
            url2 = "https://huggingface.co/renikud/renikud/resolve/main/model.onnx"
            try:
                urllib.request.urlretrieve(url2, model_path)
                print(f"✅ Модель скачана ({model_path.stat().st_size / 1024 / 1024:.2f} MB)")
            except Exception as e2:
                print(f"❌ Ошибка скачивания: {e2}")
                sys.exit(1)
    
    print("📥 Загрузка ReNikud...")
    g2p = G2P(str(model_path))
    print("✅ ReNikud загружен")
    
except ImportError:
    print("❌ ReNikud не установлен. Установите: pip install renikud-onnx")
    sys.exit(1)
except Exception as e:
    print(f"❌ Ошибка: {e}")
    sys.exit(1)

text = 'שלום אורחים יקרים! אני המערכת הדיגיטלית של מלון רויאל ים המלח. התפקיד שלי הוא להיות הכתובת המרכזית שלכם לכל בקשה, שאלה או בעיה במהלך השהות שלכם במלון. במקום לפנות נציג צוות - פשוט תדברו איתי! במה ניתן לעזור?'

print("\n📝 Исходный текст (без огласовок):")
print(text)

# Преобразование в фонемы с ударениями
print("\n🔄 Преобразование в фонемы с ударениями...")
try:
    ps = g2p.phonemize(text)
    print("\n✅ Фонемы с ударениями:")
    print(ps)
except Exception as e:
    print(f"❌ Ошибка: {e}")
    sys.exit(1)

print("\n📥 Загрузка модели Kokoro...")
device = "cpu"

try:
    km = KModel(
        repo_id="hexgrad/Kokoro-82M",
        model=str(MODEL_PATH)
    ).to(device).eval()
    print("✅ Kokoro загружен")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    sys.exit(1)

try:
    voice = torch.load(VOICE_PATH, map_location=device, weights_only=True)
    print("✅ Голос загружен")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    sys.exit(1)

print("\n🎤 Синтез речи...")
print("-" * 70)

# Разбиваем на предложения для лучшего качества
SPLIT = re.compile(r"(?<=[.!?])\s+")
pieces = []

for chunk in SPLIT.split(ps):
    if not chunk:
        continue
    n = min(len(chunk), voice.shape[0]) - 1
    with torch.no_grad():
        out = km(chunk, voice[n], speed=1.0, return_output=True)
    pieces.append(out.audio.cpu().numpy())

if not pieces:
    sys.exit("❌ Не удалось сгенерировать аудио.")

output_file = "greeting_renikud_fixed.wav"
sf.write(output_file, np.concatenate(pieces), 24000)

print(f"\n✅ Готово! Файл сохранён: {output_file}")
print(f"📁 Путь: {Path(output_file).absolute()}")
print("=" * 70)