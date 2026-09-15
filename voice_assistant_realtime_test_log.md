# Voice assistant realtime test log

Начат: 2026-09-15 14:13:12

- PASS: `51_51_smoke_and_stranger` — expected `urgent`, actual `📋 Намерение: urgent, Тема: None`.
- PASS: `52_52_room_entry_complaint` — expected `complaint`, actual `📋 Намерение: complaint, Тема: None`.
- PASS: `53_53_multiple_maintenance` — expected `request`, actual `📋 Намерение: request, Тема: None`.
- PASS: `54_54_pillow_and_pool_info` — expected `separate`, actual `📋 Намерение: separate, Тема: None`.

## Наблюдения

- Длинные записи №52 и №54 Whisper разбил на два фрагмента. Несмотря на это, №52 корректно ушла как `complaint`, а №54 — как `separate`; открытого несоответствия нет.
- В №53 DeepSeek вернул `wait_for_more: True` без точной старой фразы `האם אתם רוצים עוד משהו`, что подтвердило работу нового явного состояния диалога.
