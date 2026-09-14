# greeting.py
import os
import soundfile as sf
from pathlib import Path
from kokoro import Kokoro

# Путь к модели
MODEL_PATH = Path(__file__).resolve().parent / "kokoro_v1_hebrew.pth"
VOICE_PATH = Path(__file__).resolve().parent / "voices" / "he_shaul.pt"

# Текст с огласовками (никак без них не обойтись!)
text = "שָׁלוֹם אוֹרְחִים יְקָרִים! אֲנִי הַמַּעֲרֶכֶת הַדִּיגִיטָלִית שֶׁל מְלוֹן רוֹיַל יָם הַמֶּלַח. הַתַּפְקִיד שֶׁלִּי הוּא לִהְיוֹת הַכְּתוֹבֶת הַמֶּרְכָּזִית שֶׁלָּכֶם לְכָל בַּקָּשָׁה, שְׁאֵלָה אוֹ בְּעָיָה בְּמַהֲלָךְ הַשָּׁהוּת שֶׁלָּכֶם בַּמָּלוֹן. בִּמְקוֹם לִפְנוֹת נְצִיג צְוֶת - פַּשְׁטוּ תְּדַבְּרוּ אִיתִּי! בַּמָּה נִיתָּן לַעֲזוֹר?"

# Инициализация модели
kokoro = Kokoro(
    model_path=str(MODEL_PATH),
    voice_path=str(VOICE_PATH),
    model_type="pytorch"
)

# Синтез речи
audio, sample_rate = kokoro.create(text, voice="he_shaul")

# Сохранение
output_file = "greeting_royal_dead_sea.wav"
sf.write(output_file, audio, sample_rate)
print(f"✅ Готово! Файл сохранён: {output_file}")