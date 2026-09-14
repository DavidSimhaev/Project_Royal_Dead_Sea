# record_kokoro_files.py
import os
import sys
import re
import time
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# ⭐ ПУТИ К KOKORO
# ============================================================
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

KOKORO_BASE = Path("C:/Users/david/Desktop/kokoro-hebrew-main/inference")
KOKORO_MODEL_PATH = KOKORO_BASE / "kokoro_v1_hebrew.pth"
KOKORO_VOICE_PATH = KOKORO_BASE / "voices" / "he_shaul.pt"

# ============================================================
# ⭐ ПАПКА ДЛЯ СОХРАНЕНИЯ
# ============================================================
OUTPUT_DIR = HERE / "kokoro_audio"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# ⭐ ЗАГРУЗКА KOKORO
# ============================================================
print("📥 Загрузка Kokoro...")
device = "cpu"
try:
    sys.path.insert(0, str(KOKORO_BASE))
    from kokoro import KModel
    from hebrew_g2p import phonemize_hebrew
    
    km = KModel(repo_id="hexgrad/Kokoro-82M", model=str(KOKORO_MODEL_PATH)).to(device).eval()
    voice = torch.load(KOKORO_VOICE_PATH, map_location=device, weights_only=True)
    print("✅ Kokoro загружен")
except Exception as e:
    print(f"❌ Ошибка загрузки Kokoro: {e}")
    sys.exit(1)

# ============================================================
# ⭐ ЗАГРУЗКА RE NIKUD
# ============================================================
try:
    from renikud_onnx import G2P
    renikud_model_path = HERE / "renikud_model.onnx"
    g2p = G2P(str(renikud_model_path))
    print("✅ ReNikud загружен")
except Exception as e:
    print(f"⚠️ ReNikud не загружен: {e}")
    g2p = None

# ============================================================
# ⭐ НОВЫЕ ЗАПИСИ ДЛЯ ПЕРЕВОДА НА ПЕРСОНАЛА
# ============================================================
RECORDS = {
    "transfer_confirm": {
        "file": "transfer_confirm_kokoro.mp3",
        "description": "Подтверждение перевода на персонала",
        "text": "הבנתי. אתם רוצים שאעביר אתכם לנציג צוות המלון, נכון?"
    },
    "transfer_complete": {
        "file": "transfer_complete_kokoro.mp3",
        "description": "Перевод на персонала выполнен",
        "text": "טוב, מעביר אתכם עכשיו למרכזיה להמשך טיפול."
    }
}

# ============================================================
# 🗣️ ФУНКЦИЯ ЗАПИСИ
# ============================================================
def record_audio(text, output_file, description):
    """Записывает аудио через Kokoro"""
    print(f"🎤 ЗАПИСЬ: {description}")
    print(f"📝 Текст: {text}")
    
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
                out = km(ps, voice[n], speed=0.9, return_output=True)
            pieces.append(out.audio.cpu().numpy())
        
        if pieces:
            audio_data = np.concatenate(pieces)
            
            # ⭐ УСИЛЕНИЕ
            audio_data = audio_data * 5.0
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # ⭐ НОРМАЛИЗАЦИЯ
            max_val = np.max(np.abs(audio_data))
            if max_val > 0.01:
                audio_data = audio_data / max_val * 0.95
            
            output_path = OUTPUT_DIR / output_file
            sf.write(str(output_path), audio_data, 24000)
            
            print(f"✅ Сохранено: {output_path} ({len(audio_data) / 24000:.2f} сек)")
            return output_path
        else:
            print(f"❌ Не удалось сгенерировать аудио для: {description}")
            return None
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None

# ============================================================
# ⭐ ПРОВЕРКА НАЛИЧИЯ ФАЙЛОВ
# ============================================================
def check_existing_files():
    """Проверяет, какие файлы уже существуют"""
    existing = []
    missing = []
    
    for key, record in RECORDS.items():
        file_path = OUTPUT_DIR / record["file"]
        if file_path.exists():
            existing.append(record["file"])
        else:
            missing.append(record["file"])
    
    return existing, missing

# ============================================================
# 🚀 MAIN
# ============================================================
def main():
    print("=" * 60)
    print("🎙️ ЗАПИСЬ НОВЫХ ФАЙЛОВ ДЛЯ ПЕРЕВОДА НА ПЕРСОНАЛА")
    print("=" * 60)
    print(f"📁 Папка: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Проверяем существующие файлы
    existing, missing = check_existing_files()
    
    if existing:
        print(f"\n✅ Уже существует: {len(existing)} файлов")
        for f in existing:
            print(f"   ✓ {f}")
    
    if missing:
        print(f"\n❌ Будет создано: {len(missing)} файлов")
        for f in missing:
            print(f"   ✗ {f}")
    
    print("\n" + "=" * 60)
    
    # Спрашиваем подтверждение
    response = input("🚀 Начать запись? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Отменено")
        return
    
    total = len(RECORDS)
    success = 0
    failed = []
    
    for key, record in RECORDS.items():
        print("\n" + "-" * 60)
        result = record_audio(record["text"], record["file"], record["description"])
        if result:
            success += 1
        else:
            failed.append(record["file"])
        time.sleep(0.3)
    
    print("\n" + "=" * 60)
    print(f"✅ Готово! {success}/{total} файлов записано")
    
    if failed:
        print(f"❌ Не удалось записать: {len(failed)} файлов")
        for f in failed:
            print(f"   ✗ {f}")
    
    print(f"📁 Папка: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()