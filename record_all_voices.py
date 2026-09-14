# record_all_voices.py
"""
Создает все аудиозаписи для обоих дикторов:
- Kokoro (he_shaul, МУЖЧИНА) → папка Kokoro_RECORD/
- ElevenLabs (Jessica, ЖЕНЩИНА) → папка ElevenLabs_RECORD/

⭐ АВТОМАТИЧЕСКОЕ ПРЕОБРАЗОВАНИЕ РОДА ЧЕРЕЗ DEEPSEEK
Пользователь вводит текст в мужском роде → DeepSeek преобразует в женский

⭐ НОВОЕ: интерактивная запись отдельно для Kokoro (п.4) и ElevenLabs (п.5)
⭐ В п.5 (только ElevenLabs) перевод рода ОТКЛЮЧЕН — текст идёт как есть
"""
import os
import sys
import re
import time
import json
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

# ============================================================
# ⭐ ПУТИ К KOKORO
# ============================================================
HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
sys.path.insert(0, str(HERE))

KOKORO_BASE = Path("C:/Users/david/Desktop/kokoro-hebrew-main/inference")
KOKORO_MODEL_PATH = KOKORO_BASE / "kokoro_v1_hebrew.pth"
KOKORO_VOICE_PATH = KOKORO_BASE / "voices" / "he_shaul.pt"

# ============================================================
# ⭐ ПАПКИ ДЛЯ СОХРАНЕНИЯ
# ============================================================
KOKORO_DIR = HERE / "Kokoro_RECORD"
ELEVENLABS_DIR = HERE / "ElevenLabs_RECORD"
TEST_AUTO_REQUEST_DIR = HERE / "test_auto_request_mp3"

KOKORO_DIR.mkdir(exist_ok=True)
ELEVENLABS_DIR.mkdir(exist_ok=True)
TEST_AUTO_REQUEST_DIR.mkdir(exist_ok=True)

# ============================================================
# ⭐ НАСТРОЙКИ DEEPSEEK
# ============================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# ============================================================
# ⭐ НАСТРОЙКИ ELEVENLABS
# ============================================================
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
JESSICA_VOICE_ID = "cgSgspJ2msm6clMCkdW9"
ELEVENLABS_MODEL = "eleven_v3"

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
# ⭐ ФУНКЦИЯ ПРЕОБРАЗОВАНИЯ В ЖЕНСКИЙ РОД (ЧЕРЕЗ DEEPSEEK)
# ============================================================
def convert_to_female(text_male):
    """
    Преобразует текст из мужского рода в женский через DeepSeek.
    Возвращает женскую версию текста.
    """
    print(f"\n🔄 Преобразование в женский род...")
    print(f"   Мужской: {text_male}")
    
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_URL,
            timeout=20.0
        )

        system_prompt = """אתה מומחה בעברית. תפקידך להמיר טקסטים מלשון זכר ללשון נקבה, אבל רק עבור הדובר עצמו (גוף ראשון).

חשוב מאוד:
- שנה ONLY את הצורות שבהן הדובר מדבר על עצמו (גוף ראשון: אני/אנחנו).
- אל תשנה פנייה לקהל/למאזינים (גוף שני: אתה/אתם/אתן) - השאר אותן בלשון זכר כמו במקור.
- אל תשנה גוף שלישי (הוא/היא/הם/הן) - השאר כמו במקור.

כללים:
1. שנה רק פעלים וכינויים בגוף ראשון (אני...) מלשון זכר ללשון נקבה.
2. פניות לקהל (אתכם, אתם, לכם, רוצים, etc.) - השאר בלשון זכר.
3. אל תשנה מילים שלא צריך לשנות.
4. שמור על אותו סדר מילים ומשמעות זהה.
5. החזר רק את הטקסט המומר, בלי הסברים, בלי מרכאות, בלי הקדמות.
6. אם הטקסט כבר בלשון נקבה בגוף ראשון - החזר אותו כמו שהוא.
7. אל תוסיף או תוריד מילים.

דוגמאות:
- "אני מעביר אותכם" → "אני מעבירה אותכם" (רק "מעביר"→"מעבירה", "אותכם" נשאר)
- "אני בודק" → "אני בודקת"
- "אני רוצה" → "אני רוצה" (אין שינוי)
- "אני שמח" → "אני שמחה"
- "הבנתי. אז אתם רוצים:" → "הבנתי. אז אתם רוצים:" (אין שינוי! "הבנתי" זה גוף ראשון אבל כבר נייטרלי, "אתם רוצים" זה גוף שני)
- "בסדר גמור. אני מעביר אותכם למחלקת הספא. מאחלים לכם חופשה מהנה." → "בסדר גמור. אני מעבירה אותכם למחלקת הספא. מאחלות לכם חופשה מהנה." (רק "מעביר"→"מעבירה" ו"מאחלים"→"מאחלות" - אלו פעלים שהדובר עושה)
- "אני מצטער" → "אני מצטערת"
- "אני יכול לעזור" → "אני יכולה לעזור"
""" 
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_male}
            ],
            temperature=0.1,
            max_tokens=500,
            stream=False
        )
        
        text_female = response.choices[0].message.content.strip()
        
        # Убираем возможные кавычки
        text_female = text_female.strip('"').strip("'").strip()
        
        print(f"   Женский: {text_female}")
        return text_female
        
    except Exception as e:
        print(f"⚠️ Ошибка преобразования: {e}")
        print(f"   Использую мужской вариант как fallback")
        return text_male

# ============================================================
# ⭐ ВСЕ ЗАПИСИ (ЕДИНЫЙ СПИСОК)
# Тексты по умолчанию - МУЖСКОЙ РОД (для Kokoro)
# ============================================================
ALL_RECORDS = {
    # ═══════════════════════════════════════════════════════════
    # СЛУЖЕБНЫЕ ЗАПИСИ
    # ═══════════════════════════════════════════════════════════
    "transfer_confirm": {
        "file_kokoro": "transfer_confirm_kokoro.mp3",
        "file_eleven": "transfer_confirm.mp3",
        "description": "Подтверждение перевода на персонала",
        "text": "הבנתי אתכם. תרצו שאעביר אתכם לנציג אנושי שיוכל לעזור לכם?"
    },
    "transfer_complete": {
        "file_kokoro": "transfer_complete_kokoro.mp3",
        "file_eleven": "transfer_complete.mp3",
        "description": "Перевод на персонала выполнен",
        "text": "הפנייה מועברת כעת לנציג אנושי להמשך טיפול. שיהיה לכם חופשה מהנה."
    },
    "not_understood": {
        "file_kokoro": "not_understood_kokoro.mp3",
        "file_eleven": "not_understood.mp3",
        "description": "ИИ не понял запрос (первый раз)",
        "text": "מצטער, לא הצלחתי להבין את בקשתכם. אני מתמחה במתן מידע וסיוע בנושאים הקשורים למלון בלבד. תוכלו לנסות לנסח מחדש את בקשתכם?"
    },
    
    # ⭐ ПОДТВЕРЖДЕНИЕ ПЕРЕВОДА В СПА
    "spa_transfer_complete": {
        "file_kokoro": "spa_transfer_complete_kokoro.mp3",
        "file_eleven": "spa_transfer_complete.mp3",
        "description": "Подтверждение перевода в спа-отдел",
        "text": "בסדר גמור. אני מעביר אותכם למחלקת הספא. מאחלים לכם חופשה מהנה."
    },
    
    # ═══════════════════════════════════════════════════════════
    # ИНФОРМАЦИЯ О ТРАПЕЗАХ
    # ═══════════════════════════════════════════════════════════
    "info_breakfast": {
        "file_kokoro": "info_breakfast_kokoro.mp3",
        "file_eleven": "info_breakfast.mp3",
        "description": "Завтрак",
        "text": "ארוחת בוקר מוגשת בין 7 בבוקר עד 10 בבוקר בעולם המסעדה שבקומה ג'י"
    },
    "info_lunch": {
        "file_kokoro": "info_lunch_kokoro.mp3",
        "file_eleven": "info_lunch.mp3",
        "description": "Обед",
        "text": "ארוחת צהריים מוגשת בין 1 עד 2 וחצי בצהריים בעולם המסעדה שבקומה ג'י"
    },
    "info_dinner": {
        "file_kokoro": "info_dinner_kokoro.mp3",
        "file_eleven": "info_dinner.mp3",
        "description": "Ужин",
        "text": "ארוחת ערב מוגשת בין 6 בערב עד 9 בערב בעולם המסעדה שבקומה ג'י"
    },
    "info_all_meals": {
        "file_kokoro": "info_all_meals_kokoro.mp3",
        "file_eleven": "info_all_meals.mp3",
        "description": "Все трапезы (общая информация)",
        "text": "ארוחת בוקר בין 7 בבוקר עד 10 בבוקר. ארוחת צהריים בין 1 עד 2 וצי בצהריים. ארוחת ערב בין 6 בערב עד 9 בערב. כל הארוחות מוגשות במסעדה שבקומה ג'י"
    },
    
    # ═══════════════════════════════════════════════════════════
    # ИНФОРМАЦИЯ О ЛОББИ
    # ═══════════════════════════════════════════════════════════
    "info_lobby": {
        "file_kokoro": "info_lobby_kokoro.mp3",
        "file_eleven": "info_lobby.mp3",
        "description": "Лобби",
        "text": "לובי מלון פתוח מדי יום בין השעות 11:30 עד 23 בלילה והוא ממוקם בקומה e. אתם מוזמנים לשהות בו לאורך כל שעות הפעילות"
    },
}

# ============================================================
# 🗣️ ФУНКЦИЯ ЗАПИСИ KOKORO
# ============================================================
def record_kokoro(text, output_file, description):
    """Записывает аудио через Kokoro (МУЖСКОЙ голос)"""
    print(f"🎤 KOKORO (мужской): {description}")
    
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
                out = km(ps, voice[n], speed=0.9, return_output=True)
            pieces.append(out.audio.cpu().numpy())
        
        if pieces:
            audio_data = np.concatenate(pieces)
            audio_data = audio_data * 5.0
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            max_val = np.max(np.abs(audio_data))
            if max_val > 0.01:
                audio_data = audio_data / max_val * 0.95
            
            output_path = KOKORO_DIR / output_file
            sf.write(str(output_path), audio_data, 24000)
            
            print(f"✅ Сохранено: {output_path.name} ({len(audio_data) / 24000:.2f} сек)")
            return output_path
        else:
            print(f"❌ Не удалось сгенерировать аудио")
            return None
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None

# ============================================================
# 🗣️ ФУНКЦИЯ ЗАПИСИ ELEVENLABS
# ============================================================
def record_elevenlabs(text, output_file, description):
    """Записывает аудио через ElevenLabs (ЖЕНСКИЙ голос)"""
    print(f"🎤 ELEVENLABS (женский): {description}")
    
    try:
        import requests
        
        clean_text = text.replace('"', '').replace("'", "").replace("\n", " ").strip()[:500]
        
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
            },
            "language_code": "he"
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            output_path = ELEVENLABS_DIR / output_file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Сохранено: {output_path.name}")
            return output_path
        else:
            print(f"❌ Ошибка ElevenLabs: {response.status_code} - {response.text[:100]}")
            return None
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None

# ============================================================
# 🧪 АВТОМАТИЧЕСКИЕ ТЕСТОВЫЕ ЗАПРОСЫ ДЛЯ МИКРОФОНА
# ============================================================
AUTO_TEST_REQUESTS = {
    "towels_to_room_kokoro.mp3": "אני רוצה מגבות לחדר",
}

def create_auto_test_requests():
    """Создаёт готовые Kokoro-запросы для самотеста voice_assistant_v2.py."""
    print("\n" + "=" * 60)
    print("🧪 СОЗДАНИЕ АВТОМАТИЧЕСКИХ ТЕСТОВЫХ ЗАПРОСОВ")
    print(f"📁 Папка: {TEST_AUTO_REQUEST_DIR}")
    print("=" * 60)
    created = []
    for filename, text in AUTO_TEST_REQUESTS.items():
        print(f"\n📝 {text}")
        try:
            text_with_niqqud = g2p.phonemize(text) if g2p else text
            pieces = []
            for chunk in re.split(r"(?<=[.!?])\\s+", text_with_niqqud.strip()):
                if not chunk:
                    continue
                phonemes, _ = phonemize_hebrew(chunk)
                if not phonemes:
                    continue
                n = min(len(phonemes), voice.shape[0]) - 1
                with torch.no_grad():
                    output = km(phonemes, voice[n], speed=0.9, return_output=True)
                pieces.append(output.audio.cpu().numpy())
            if not pieces:
                print("❌ Kokoro не создал аудио")
                continue
            audio_data = np.clip(np.concatenate(pieces) * 5.0, -1.0, 1.0)
            peak = np.max(np.abs(audio_data))
            if peak > 0.01:
                audio_data = audio_data / peak * 0.95
            output_path = TEST_AUTO_REQUEST_DIR / filename
            sf.write(str(output_path), audio_data, 24000)
            created.append(output_path)
            print(f"✅ Сохранено: {output_path}")
        except Exception as e:
            print(f"❌ Ошибка создания {filename}: {e}")
    print(f"\n✅ Создано тестовых записей: {len(created)}")
    return created

# ============================================================
# ⭐ УНИВЕРСАЛЬНЫЙ ИНТЕРАКТИВНЫЙ РЕЖИМ
# ============================================================
def interactive_add_records(do_kokoro=True, do_eleven=True, convert_gender=True):
    """
    Интерактивный режим добавления записей.
    Пользователь вводит ТОЛЬКО мужской текст.
    
    do_kokoro      - записывать ли Kokoro (мужской голос)
    do_eleven      - записывать ли ElevenLabs (женский голос)
    convert_gender - преобразовывать ли текст в женский род через DeepSeek
                     (True  - для пунктов 2 и 5 с авто-переводом... но см. ниже)
                     (False - текст идёт как есть, без изменений)
    """
    # ⭐ Определяем режим
    if do_kokoro and do_eleven:
        mode_title = "ОБА ДИКТОРА"
        mode_desc = "Kokoro (мужской) + ElevenLabs (женский)"
    elif do_kokoro:
        mode_title = "ТОЛЬКО KOKORO"
        mode_desc = "Kokoro (мужской голос)"
    else:
        mode_title = "ТОЛЬКО ELEVENLABS"
        if convert_gender:
            mode_desc = "ElevenLabs (женский голос, авто-перевод через DeepSeek)"
        else:
            mode_desc = "ElevenLabs (женский голос, БЕЗ перевода рода — текст как есть)"
    
    print("\n" + "=" * 60)
    print(f"✍️ ИНТЕРАКТИВНОЕ ДОБАВЛЕНИЕ ЗАПИСЕЙ — {mode_title}")
    print("=" * 60)
    print(f"🎯 Режим: {mode_desc}")
    print()
    if do_eleven and convert_gender:
        print("💡 Вводите текст в МУЖСКОМ роде")
        print("   Женский вариант создастся автоматически через DeepSeek.")
    elif do_eleven and not convert_gender:
        print("💡 Вводите текст КАК ЕСТЬ (в любом роде)")
        print("   Текст пойдёт в ElevenLabs без изменений.")
    else:
        print("💡 Вводите текст в МУЖСКОМ роде (для Kokoro)")
    print()
    print("Для выхода введите 'exit' в поле имени файла")
    print("=" * 60)
    
    added_records = []
    
    while True:
        print("\n" + "-" * 60)
        
        # ⭐ ВВОД ИМЕНИ ФАЙЛА
        file_name = input("📁 Имя файла (без .mp3) или 'exit': ").strip()
        
        if file_name.lower() == 'exit':
            print("\n👋 Выход из режима добавления")
            break
        
        if not file_name:
            print("⚠️ Имя файла не может быть пустым")
            continue
        
        # Убираем .mp3 если пользователь случайно добавил
        if file_name.endswith('.mp3'):
            file_name = file_name[:-4]
        
        # Формируем имена для обоих дикторов
        kokoro_file = f"{file_name}_kokoro.mp3"
        eleven_file = f"{file_name}.mp3"
        
        # ⭐ Показываем, что будет создано
        print(f"\n📝 Будет создано:")
        if do_kokoro:
            print(f"   🎤 Kokoro (МУЖ):     {kokoro_file}")
        if do_eleven:
            print(f"   🎤 ElevenLabs (ЖЕН): {eleven_file}")
        
        # ⭐ ВВОД ТЕКСТА
        if do_eleven and not convert_gender:
            print("\n📝 Введите текст на иврите КАК ЕСТЬ:")
        else:
            print("\n📝 Введите текст на иврите В МУЖСКОМ РОДЕ:")
        print("   (для завершения ввода введите пустую строку)")
        
        lines = []
        while True:
            line = input("   > ").strip()
            if not line:
                break
            lines.append(line)
        
        text_male = " ".join(lines).strip()
        
        if not text_male:
            print("⚠️ Текст не может быть пустым")
            continue
        
        print(f"\n📝 Введённый текст: {text_male}")
        
        # ⭐ ПРЕОБРАЗОВАНИЕ В ЖЕНСКИЙ РОД (только если convert_gender=True)
        if do_eleven and convert_gender:
            text_female = convert_to_female(text_male)
        elif do_eleven and not convert_gender:
            # ⭐ Без преобразования — текст идёт как есть
            text_female = text_male
            print(f"   ⏩ Перевод рода отключён — текст пойдёт как есть")
        else:
            text_female = None
        
        # ⭐ ПОДТВЕРЖДЕНИЕ
        print(f"\n📋 ИТОГОВЫЕ ТЕКСТЫ:")
        if do_kokoro:
            print(f"   👨 Kokoro:     {text_male}")
        if do_eleven:
            print(f"   👩 ElevenLabs: {text_female}")
        
        confirm = input("\n🚀 Создать записи? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Отменено")
            continue
        
        # ⭐ СОЗДАЕМ ЗАПИСИ
        print("\n" + "=" * 60)
        
        kokoro_result = None
        eleven_result = None
        
        if do_kokoro:
            kokoro_result = record_kokoro(text_male, kokoro_file, f"Интерактивная: {file_name}")
        
        if do_eleven:
            eleven_result = record_elevenlabs(text_female, eleven_file, f"Интерактивная: {file_name}")
        
        # ⭐ СОХРАНЯЕМ В МАНИФЕСТ
        if kokoro_result or eleven_result:
            added_records.append({
                "name": file_name,
                "file_kokoro": kokoro_file if do_kokoro else None,
                "file_eleven": eleven_file if do_eleven else None,
                "text_male": text_male,
                "text_female": text_female,
                "gender_converted": convert_gender if do_eleven else False,
                "kokoro_created": kokoro_result is not None,
                "eleven_created": eleven_result is not None,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            print("\n" + "=" * 60)
            print("✅ ЗАПИСИ СОЗДАНЫ:")
            if kokoro_result:
                print(f"   ✓ {kokoro_result.name}")
            if eleven_result:
                print(f"   ✓ {eleven_result.name}")
            print("=" * 60)
        else:
            print("\n❌ Не удалось создать записи")
    
    # ⭐ СОХРАНЯЕМ СПИСОК ДОБАВЛЕННЫХ В JSON
    if added_records:
        added_file = HERE / "added_records.json"
        
        existing = []
        if added_file.exists():
            try:
                with open(added_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except:
                existing = []
        
        existing.extend(added_records)
        
        with open(added_file, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=4)
        
        print(f"\n📋 Добавлено записей: {len(added_records)}")
        print(f"📋 Сохранено в: {added_file}")
    else:
        print("\n📋 Новых записей не добавлено")

# ============================================================
# ⭐ ПОКАЗ СУЩЕСТВУЮЩИХ ФАЙЛОВ
# ============================================================
def show_existing_files():
    """Показывает все существующие mp3 файлы"""
    print("\n" + "=" * 60)
    print("📁 СУЩЕСТВУЮЩИЕ ФАЙЛЫ")
    print("=" * 60)
    
    print("\n🎤 Kokoro_RECORD/:")
    kokoro_files = sorted(KOKORO_DIR.glob("*.mp3"))
    if kokoro_files:
        for f in kokoro_files:
            print(f"   ✅ {f.name}")
    else:
        print("   ⚠️ Папка пуста")
    
    print("\n🎤 ElevenLabs_RECORD/:")
    eleven_files = sorted(ELEVENLABS_DIR.glob("*.mp3"))
    if eleven_files:
        for f in eleven_files:
            print(f"   ✅ {f.name}")
    else:
        print("   ⚠️ Папка пуста")

# ============================================================
# ⭐ СОХРАНЕНИЕ МАНИФЕСТА
# ============================================================
def save_records_manifest():
    """Сохраняет манифест всех записей"""
    manifest = {
        "version": "2.4",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "records": {}
    }
    
    for key, record in ALL_RECORDS.items():
        manifest["records"][key] = {
            "kokoro_file": record["file_kokoro"],
            "eleven_file": record["file_eleven"],
            "description": record["description"],
            "text": record["text"],
            "kokoro_exists": (KOKORO_DIR / record["file_kokoro"]).exists(),
            "eleven_exists": (ELEVENLABS_DIR / record["file_eleven"]).exists()
        }
    
    manifest_path = HERE / "records_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=4)
    
    print(f"\n📋 Манифест сохранен: {manifest_path}")

# ============================================================
# 🚀 MAIN
# ============================================================
def main():
    print("=" * 60)
    print("🎙️ СОЗДАНИЕ АУДИОЗАПИСЕЙ ДЛЯ ДВУХ ДИКТОРОВ")
    print("=" * 60)
    print(f"👨 Kokoro:     {KOKORO_DIR} (мужской голос)")
    print(f"👩 ElevenLabs: {ELEVENLABS_DIR} (женский голос)")
    print("=" * 60)
    
    show_existing_files()
    
    print("\n" + "=" * 60)
    print("🚀 ВЫБЕРИТЕ ДЕЙСТВИЕ:")
    print("=" * 60)
    print("   1 - Записать базовые записи (Kokoro + ElevenLabs)")
    print("   2 - ✍️ ИНТЕРАКТИВНОЕ ДОБАВЛЕНИЕ (оба диктора, авто-перевод рода)")
    print("   3 - Только показать существующие файлы")
    print("   4 - 🎤 ИНТЕРАКТИВНО: ТОЛЬКО Kokoro (мужской голос)")
    print("   5 - 🎤 ИНТЕРАКТИВНО: ТОЛЬКО ElevenLabs (БЕЗ перевода рода)")
    print("   6 - 🧪 Создать тестовый запрос Kokoro для самотеста")
    print("=" * 60)
    
    choice = input("Ваш выбор (1/2/3/4/5/6): ").strip()

    if choice == '6':
        create_auto_test_requests()
        return
    
    # ⭐ ИНТЕРАКТИВ: ОБА ДИКТОРА (с авто-переводом рода)
    if choice == '2':
        interactive_add_records(do_kokoro=True, do_eleven=True, convert_gender=True)
        save_records_manifest()
        return
    
    # ⭐ ТОЛЬКО ПОКАЗАТЬ ФАЙЛЫ
    if choice == '3':
        save_records_manifest()
        return
    
    # ⭐ ИНТЕРАКТИВ: ТОЛЬКО KOKORO (без перевода — он и не нужен)
    if choice == '4':
        interactive_add_records(do_kokoro=True, do_eleven=False, convert_gender=False)
        save_records_manifest()
        return
    
    # ⭐ ИНТЕРАКТИВ: ТОЛЬКО ELEVENLABS (БЕЗ перевода рода!)
    if choice == '5':
        interactive_add_records(do_kokoro=False, do_eleven=True, convert_gender=False)
        save_records_manifest()
        return
    
    # ⭐ БАЗОВЫЕ ЗАПИСИ (пункт 1)
    if choice == '1':
        total = len(ALL_RECORDS)
        success_kokoro = 0
        success_eleven = 0
        failed_kokoro = []
        failed_eleven = []
        
        print(f"\n📊 Всего записей: {total}")
        print("=" * 60)
        
        for key, record in ALL_RECORDS.items():
            print("\n" + "-" * 60)
            print(f"📌 {record['description']}")
            
            text_male = record["text"]
            text_female = convert_to_female(text_male)
            
            # KOKORO
            result = record_kokoro(
                text_male,
                record["file_kokoro"],
                f"{record['description']} (мужской)"
            )
            if result:
                success_kokoro += 1
            else:
                failed_kokoro.append(record["file_kokoro"])
            time.sleep(0.3)
            
            # ELEVENLABS
            result = record_elevenlabs(
                text_female,
                record["file_eleven"],
                f"{record['description']} (женский)"
            )
            if result:
                success_eleven += 1
            else:
                failed_eleven.append(record["file_eleven"])
            time.sleep(1.0)
        
        print("\n" + "=" * 60)
        print("📊 ИТОГИ:")
        print("=" * 60)
        print(f"👨 Kokoro (мужской):     {success_kokoro}/{total}")
        if failed_kokoro:
            print(f"   ❌ Не удалось: {len(failed_kokoro)}")
            for f in failed_kokoro:
                print(f"      ✗ {f}")
        
        print(f"👩 ElevenLabs (женский): {success_eleven}/{total}")
        if failed_eleven:
            print(f"   ❌ Не удалось: {len(failed_eleven)}")
            for f in failed_eleven:
                print(f"      ✗ {f}")
        
        save_records_manifest()
        return
    
    print("❌ Неверный выбор")

if __name__ == "__main__":
    main()
