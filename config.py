# config.py
import os
from dotenv import load_dotenv

# Загружаем переменные окружения из файла .env, если он существует
load_dotenv()

# --- Корневая директория проекта ---
# config.py находится в корневой директории проекта
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- API ключи и модели ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_PLACEHOLDER")  # ИЗМЕНИТЕ ЭТО или задайте переменную окружения
if not GEMINI_API_KEY or "YOUR_API_KEY" in GEMINI_API_KEY or ("AIzaSy" not in GEMINI_API_KEY):
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! ВНИМАНИЕ: API ключ Gemini отсутствует или недействителен.  !!!")
    print("!!! Пожалуйста, задайте переменную окружения GEMINI_API_KEY   !!!")
    print("!!! или обновите значение по умолчанию в config.py.            !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    # Можно рассмотреть вариант вызова ошибки, если ключ обязателен:
    # raise ValueError("Gemini API Key is missing or invalid.")

EMBEDDING_MODEL_NAME = "DeepPavlov/rubert-base-cased-sentence"
LLM_MODEL_NAME = "gemini-2.0-flash"  # Или выбранная вами модель

# --- Настройка прокси ---
# Укажите URL прокси здесь через переменную окружения или жестко закодированное значение
PROXY_URL = os.getenv("PROXY_URL", None)  # Установите в None или "" для отключения

# --- Каталоги ---
LESSONS_DIR = os.path.join(PROJECT_ROOT, "lessons")
TXT_DIR = os.path.join(PROJECT_ROOT, "txt")
EMBEDDING_DIR = os.path.join(PROJECT_ROOT, "embeddings")
UPLOAD_FOLDER = LESSONS_DIR  # Настройка загрузки для Flask
PROMPT_DIR = os.path.join(PROJECT_ROOT, "prompts")

# --- Настройки поиска RAG ---
BM25_SCORE_THRESHOLD = 7.0
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
MAX_CHUNKS_PER_TYPE = 2
TOP_K = 4  # Первоначальный поиск для BM25 до применения пороговых значений/ограничений
FINAL_TOP_K = 5  # Не используется напрямую в текущей логике объединения, но оставлен для потенциального будущего использования

# --- Настройки генерации LLM ---
generation_config = {"temperature": 0.0}

# --- Настройки Flask и сервера ---
API_PREFIX = "/api/v2"
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 МБ
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'md', 'py', 'js', 'html', 'css', 'zip', 'docx', 'pptx', 'mp4', 'srt'}
REINDEX_SECRET = os.getenv("REINDEX_SECRET", "your-very-secret-reindex-key")  # ИЗМЕНИТЕ ЭТО или задайте переменную окружения
if REINDEX_SECRET == "your-very-secret-reindex-key":
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! ВНИМАНИЕ: Используется значение REINDEX_SECRET по умолчанию. Измените его!  !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

# --- Фоновая обработка и кэш ---
MAX_WORKERS = 4  # Для ThreadPoolExecutor
REQUEST_CACHE_TTL = 300  # 5 минут
MAX_CACHE_SIZE = 1000

# --- Имена файлов шаблонов ---
SYSTEM_FILTER_PROMPT_FILE = os.path.join(PROMPT_DIR, "system_filter_prompt.txt")
FINAL_ANSWER_PROMPT_FILE = os.path.join(PROMPT_DIR, "final_answer_prompt.txt")