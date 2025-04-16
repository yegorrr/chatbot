
# utils.py
import re
import os
import numpy as np
import uuid

# --- Очистка и обработка текста ---
def clean_text(text):
    """Базовая очистка текста для подсказок LLM и извлечённого текста."""
    if not isinstance(text, str): text = str(text)
    if isinstance(text, bytes):
        text = text.decode('utf-8', 'replace')
    text = ' '.join(text.split())  # Нормализация пробелов
    return text

def clean_extracted_text(text):
    """Удаляет избыточные пробелы и выполняет базовую очистку извлечённого текста."""
    if not text: return ""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text

def truncate_text(text, num_words=1000):
    """Обрезает текст до указанного максимального количества слов."""
    if not isinstance(text, str): return ""
    return " ".join(text.split()[:num_words])

def tokenize(text):
    """Простая токенизация слов для BM25."""
    if not isinstance(text, str): return []
    return re.findall(r'\b\w+\b', text.lower(), flags=re.UNICODE)

# --- Математика и вычисление схожести ---
def cosine_similarity(a, b):
    """Вычисляет косинусное сходство между двумя numpy-массивами."""
    if a is None or b is None: return 0.0
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray): return 0.0
    if a.ndim > 1: a = a.flatten()
    if b.ndim > 1: b = b.flatten()
    if a.shape[0] == 0 or b.shape[0] == 0: return 0.0  # Обработка пустых векторов
    if a.shape != b.shape:
        # Может происходить при разных размерностях выходов моделей или ошибках пользователя.
        # Логировать при частых случаях. Пока возвращаем 0.
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    epsilon = 1e-8
    if norm_a < epsilon or norm_b < epsilon: return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

# --- Работа с файлами и их санитизация ---
def sanitize_filename_ru(filename):
    """
    Пользовательская функция очистки имени файла, допускающая русские (кириллические) символы,
    пробелы, цифры, дефисы, подчёркивания и точки, удаляя при этом
    потенциально опасные символы и элементы обхода путей.
    Обрабатывает None и нестроковые входные данные.
    """
    if not isinstance(filename, str):
        filename = str(filename)

    allowed_chars = r'[^\w\u0400-\u04FF\-\.\s]'
    sanitized = re.sub(allowed_chars, '', filename, flags=re.UNICODE)
    sanitized = re.sub(r'[\s_-]+', '_', sanitized)
    sanitized = sanitized.strip('._- ')
    sanitized = sanitized.replace('../', '_').replace('..\\', '_')
    sanitized = sanitized.replace('/', '_').replace('\\', '_')

    max_len = 200
    if len(sanitized) > max_len:
        name, ext = os.path.splitext(sanitized)
        limit = max_len - (len(ext) + 1) if ext else max_len
        sanitized = name[:limit] + ext if ext else name[:limit]

    if not sanitized or sanitized.strip('.') == '':
        return f"sanitized_empty_{uuid.uuid4().hex[:8]}"

    return sanitized

def allowed_file(filename, allowed_extensions):
    """Проверяет, разрешено ли расширение файла."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def isADSIdentifier(filename):
    """Проверяет, содержит ли имя файла идентификатор Alternate Data Stream (Windows)."""
    return (':' in filename or '' in filename) and "Zone.Identifier" in filename

# --- Загрузка шаблонов подсказок ---
def load_prompt_template(filepath):
    """Загружает текстовый файл для использования в качестве шаблона подсказки."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"!!! ERROR: Prompt template file not found: {filepath} !!!")
        return f"ERROR: Prompt file '{os.path.basename(filepath)}' not found."
    except Exception as e:
        print(f"!!! ERROR: Failed to read prompt template {filepath}: {e} !!!")
        return f"ERROR: Failed to read prompt '{os.path.basename(filepath)}'."
