# clear_queue.py
"""
Очистка очереди отправки WhatsApp.
Показывает текущее содержимое и позволяет очистить.
"""
import json
from pathlib import Path
import os
import sys
import shutil
from datetime import datetime

# ============================================================
# ⭐ ПУТИ
# ============================================================
QUEUE_FILE = Path(r"C:\Users\david\Desktop\WhatsApp\Project\send_queue.json")
BACKUP_DIR = Path(r"C:\Users\david\Desktop\WhatsApp\Project\queue_backups")

# ============================================================
# ⭐ ЗАГРУЗКА ОЧЕРЕДИ
# ============================================================
def load_queue():
    if QUEUE_FILE.exists():
        try:
            with open(QUEUE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Ошибка чтения: {e}")
            return []
    return []

# ============================================================
# ⭐ ПОКАЗ ОЧЕРЕДИ
# ============================================================
def show_queue(queue):
    if not queue:
        print("📭 Очередь пуста")
        return
    
    print(f"\n📋 В очереди {len(queue)} сообщений:\n")
    print("=" * 60)
    
    for i, item in enumerate(queue, 1):
        msg_id = item.get("id", "?")
        text = item.get("text", "")[:80].replace("\n", " ")
        audio = item.get("audio_file")
        created = item.get("created_at", 0)
        
        # Форматируем дату
        try:
            dt = datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M:%S")
        except:
            dt = "?"
        
        print(f"#{i} [ID: {msg_id}]")
        print(f"   📅 {dt}")
        print(f"   📝 {text}...")
        if audio:
            print(f"   🎵 Аудио: {Path(audio).name}")
        print("-" * 60)

# ============================================================
# ⭐ БЭКАП ОЧЕРЕДИ
# ============================================================
def backup_queue():
    """Создаёт бэкап очереди перед очисткой"""
    if not QUEUE_FILE.exists():
        return None
    
    BACKUP_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = BACKUP_DIR / f"send_queue_backup_{timestamp}.json"
    
    try:
        shutil.copy2(QUEUE_FILE, backup_file)
        print(f"💾 Бэкап создан: {backup_file}")
        return backup_file
    except Exception as e:
        print(f"⚠️ Ошибка создания бэкапа: {e}")
        return None

# ============================================================
# ⭐ ОЧИСТКА ОЧЕРЕДИ
# ============================================================
def clear_queue(make_backup=True):
    """
    Очищает очередь.
    make_backup - создавать ли бэкап перед очисткой
    """
    if not QUEUE_FILE.exists():
        print("📭 Очередь не существует (файл не найден)")
        return True
    
    # ⭐ Бэкап
    if make_backup:
        backup_queue()
    
    # ⭐ Записываем пустой массив
    try:
        with open(QUEUE_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=4)
        print("✅ Очередь очищена")
        return True
    except Exception as e:
        print(f"❌ Ошибка очистки: {e}")
        return False

# ============================================================
# ⭐ МЕНЮ
# ============================================================
def main():
    print("=" * 60)
    print("🧹 ОЧИСТКА ОЧЕРЕДИ ОТПРАВКИ WHATSAPP")
    print("=" * 60)
    print(f"📁 Файл: {QUEUE_FILE}")
    print("=" * 60)
    
    # Проверка существования
    if not QUEUE_FILE.exists():
        print(f"\n⚠️ Файл очереди не найден: {QUEUE_FILE}")
        print("   Возможно, программа ещё не создала очередь.")
        
        # Предложить создать пустой
        create = input("\n🚀 Создать пустой файл очереди? (y/n): ").strip().lower()
        if create == 'y':
            try:
                with open(QUEUE_FILE, "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=4)
                print(f"✅ Создан пустой файл: {QUEUE_FILE}")
            except Exception as e:
                print(f"❌ Ошибка: {e}")
        return
    
    # Загружаем и показываем
    queue = load_queue()
    show_queue(queue)
    
    # Если очередь пуста - выходим
    if not queue:
        print("\n✅ Очередь уже пуста. Нечего очищать.")
        return
    
    # Меню
    print("\n" + "=" * 60)
    print("🚀 ВЫБЕРИТЕ ДЕЙСТВИЕ:")
    print("=" * 60)
    print(f"   1 - Очистить очередь (с бэкапом)")
    print(f"   2 - Очистить очередь (БЕЗ бэкапа)")
    print(f"   3 - Только показать очередь (не очищать)")
    print(f"   4 - Удалить последнее сообщение")
    print(f"   5 - Удалить по ID")
    print("=" * 60)
    
    choice = input("Ваш выбор (1/2/3/4/5): ").strip()
    
    if choice == '1':
        confirm = input(f"⚠️ Удалить все {len(queue)} сообщений? (y/n): ").strip().lower()
        if confirm == 'y':
            clear_queue(make_backup=True)
        else:
            print("❌ Отменено")
    
    elif choice == '2':
        confirm = input(f"⚠️ Удалить все {len(queue)} сообщений БЕЗ бэкапа? (y/n): ").strip().lower()
        if confirm == 'y':
            clear_queue(make_backup=False)
        else:
            print("❌ Отменено")
    
    elif choice == '3':
        print("\n✅ Очередь не тронута")
        return
    
    elif choice == '4':
        # Удалить последнее
        last = queue[-1]
        print(f"\n🗑️ Последнее сообщение:")
        print(f"   ID: {last.get('id')}")
        print(f"   Текст: {last.get('text', '')[:100]}...")
        
        confirm = input("\nУдалить последнее? (y/n): ").strip().lower()
        if confirm == 'y':
            backup_queue()
            queue.pop()
            try:
                with open(QUEUE_FILE, "w", encoding="utf-8") as f:
                    json.dump(queue, f, ensure_ascii=False, indent=4)
                print(f"✅ Удалено. Осталось: {len(queue)}")
            except Exception as e:
                print(f"❌ Ошибка: {e}")
        else:
            print("❌ Отменено")
    
    elif choice == '5':
        # Удалить по ID
        target_id = input("Введите ID для удаления: ").strip()
        
        found = False
        for item in queue:
            if str(item.get("id")) == target_id:
                print(f"\n🗑️ Найдено сообщение:")
                print(f"   ID: {item.get('id')}")
                print(f"   Текст: {item.get('text', '')[:100]}...")
                found = True
                break
        
        if found:
            confirm = input("\nУдалить это сообщение? (y/n): ").strip().lower()
            if confirm == 'y':
                backup_queue()
                queue = [item for item in queue if str(item.get("id")) != target_id]
                try:
                    with open(QUEUE_FILE, "w", encoding="utf-8") as f:
                        json.dump(queue, f, ensure_ascii=False, indent=4)
                    print(f"✅ Удалено. Осталось: {len(queue)}")
                except Exception as e:
                    print(f"❌ Ошибка: {e}")
            else:
                print("❌ Отменено")
        else:
            print(f"❌ Сообщение с ID {target_id} не найдено")
    
    else:
        print("❌ Неверный выбор")
    
    # Финальная проверка
    print("\n" + "=" * 60)
    final_queue = load_queue()
    print(f"📊 Сообщений в очереди: {len(final_queue)}")
    print("=" * 60)

if __name__ == "__main__":
    main()