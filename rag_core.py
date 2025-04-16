# rag_core.py
import os
import glob
import re
import numpy as np
import time
import fitz  # PyMuPDF
import docx  # python-docx
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Импорт из локальных модулей
import config
from utils import clean_extracted_text, tokenize, cosine_similarity, isADSIdentifier

# --- Глобальные ресурсы RAG ---
# Они будут инициализированы с помощью initialize_rag_resources(), вызываемой из app.py
semantic_model = None
BM25_INDEX = None
CHUNK_FILES = []      # Список абсолютных путей к файлам .txt
CHUNK_TEXTS = []      # Список текстового содержимого, соответствующего CHUNK_FILES
CHUNK_TOKENS = []     # Список токенизированного текста для BM25
CHUNK_EMBEDDINGS = [] # Список numpy-эмбеддингов (или None, если отсутствует/не удалось создать)

# === Извлечение текста ===
def extract_text(filepath):
    """
    Извлекает текст из поддерживаемых типов файлов (docx, txt, pdf).
    Очищает полученный текст.
    """
    _, extension = os.path.splitext(filepath.lower())
    text = None
    try:
        if extension == ".docx":
            doc = docx.Document(filepath)
            full_text = [para.text for para in doc.paragraphs]
            text = '\n'.join(full_text)
        elif extension == ".txt":
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        elif extension == ".pdf":
            try:
                doc = fitz.open(filepath)
                full_text = ""
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    page_text = page.get_text("text")
                    if page_text:
                        full_text += page_text + "\n"
                doc.close()
                text = full_text
            except Exception as pdf_err:
                print(f"      ERROR processing PDF file {filepath} with PyMuPDF: {pdf_err}")
                try:
                    if 'doc' in locals() and doc:
                        doc.close()
                except:
                    pass
                text = None
        else:
            return None  # Пропускаем неподдерживаемые типы

        if text:
            cleaned_text = clean_extracted_text(text)
            return cleaned_text if cleaned_text else None
        else:
            return None
    except Exception as e:
        print(f"    ERROR during text extraction process for {filepath}: {e}")
        return None

# === Генерация эмбеддингов ===
def generate_embedding(text):
    """Генерирует эмбеддинг для заданного текста с использованием глобальной семантической модели."""
    global semantic_model
    if semantic_model is None:
        print("ERROR: Semantic model not loaded, cannot generate embedding.")
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        embedding = semantic_model.encode(text)
        if isinstance(embedding, np.ndarray) and embedding.ndim > 1:
            embedding = embedding.flatten()
        elif not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding).flatten()
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# === Построение индекса ===
def build_bm25_index(token_list):
    """Вспомогательная функция для безопасного создания индекса BM25."""
    valid_tokens = [tokens for tokens in token_list if tokens]
    if not valid_tokens:
        print("    WARNING: No valid token lists found for BM25 index construction.")
        return None
    try:
        index = BM25Okapi(valid_tokens)
        print(f"    BM25 index built successfully from {len(valid_tokens)} documents.")
        return index
    except Exception as e:
        print(f"    ERROR building BM25 index: {e}")
        return None

# === Основная функция загрузки данных и построения индекса ===
def rebuild_rag_index():
    """
    Выполняет предварительную обработку (извлечение текста, генерация эмбеддингов для новых/отсутствующих файлов)
    и затем перезагружает все данные RAG из каталогов TXT_DIR и EMBEDDING_DIR.
    Обновляет глобальные переменные. ДОЛЖНО вызываться после загрузки semantic_model.
    Возвращает кортеж: (True/False, сообщение)
    """
    global BM25_INDEX, CHUNK_FILES, CHUNK_TEXTS, CHUNK_TOKENS, CHUNK_EMBEDDINGS

    if semantic_model is None:
        return False, "Cannot rebuild index: Semantic model not loaded."

    print("\n--- Starting RAG Preprocessing & Re-indexing Process ---")
    start_time = time.time()
    preprocessing_errors = []
    preprocessing_success_count = {"txt": 0, "npy": 0}

    # === Шаг 1: Цикл предварительной обработки (сканирование LESSONS_DIR) ===
    print(f"  Scanning source directory ({config.LESSONS_DIR}) for files to process...")
    source_files_processed = 0
    try:
        # Итерация по всем файлам в подкаталогах LESSONS_DIR
        for lesson_folder_path in glob.glob(os.path.join(config.LESSONS_DIR, "*")):
            if not os.path.isdir(lesson_folder_path):
                continue

            lesson_folder_name = os.path.basename(lesson_folder_path)

            for filename in os.listdir(lesson_folder_path):
                lesson_file_path = os.path.join(lesson_folder_path, filename)
                if not os.path.isfile(lesson_file_path) or isADSIdentifier(filename):
                    continue

                source_files_processed += 1
                base_filename_no_ext, source_extension = os.path.splitext(filename)
                source_extension = source_extension.lower()

                # Определяем соответствующие пути
                txt_subfolder = os.path.join(config.TXT_DIR, lesson_folder_name)
                npy_subfolder = os.path.join(config.EMBEDDING_DIR, lesson_folder_name)
                txt_file_path = os.path.join(txt_subfolder, base_filename_no_ext + ".txt")
                npy_file_path = os.path.join(npy_subfolder, base_filename_no_ext + ".npy")

                extracted_text = None
                txt_created_or_exists = False

                # 1a. Проверка/создание TXT файла
                if not os.path.exists(txt_file_path):
                    extracted_text = extract_text(lesson_file_path)
                    if extracted_text:
                        try:
                            os.makedirs(txt_subfolder, exist_ok=True)
                            with open(txt_file_path, "w", encoding="utf-8") as f:
                                f.write(extracted_text)
                            preprocessing_success_count["txt"] += 1
                            txt_created_or_exists = True
                        except Exception as e:
                            err_msg = f"Failed to write TXT file {txt_file_path}: {e}"
                            print(f"        ERROR: {err_msg}")
                            preprocessing_errors.append(f"{filename}: {err_msg}")
                            if os.path.exists(txt_file_path):
                                os.remove(txt_file_path)
                    else:
                        if source_extension in ['.docx', '.txt', '.pdf']:
                            err_msg = f"Failed to extract text from {filename}"
                            if source_extension != '.pdf':
                                preprocessing_errors.append(f"{filename}: {err_msg}")
                else:
                    txt_created_or_exists = True

                # 1b. Проверка/создание NPY файла (только если TXT существует или создан)
                if txt_created_or_exists and not os.path.exists(npy_file_path):
                    if extracted_text is None:  # Загружаем, если не было получено ранее
                        try:
                            with open(txt_file_path, 'r', encoding='utf-8') as f:
                                extracted_text = f.read().strip()
                        except Exception as e:
                            err_msg = f"Failed to read existing TXT {txt_file_path} for embedding: {e}"
                            print(f"        ERROR: {err_msg}")
                            preprocessing_errors.append(f"{filename}: {err_msg}")
                            extracted_text = None

                    if extracted_text and semantic_model:
                        embedding = generate_embedding(extracted_text)
                        if embedding is not None:
                            try:
                                os.makedirs(npy_subfolder, exist_ok=True)
                                np.save(npy_file_path, embedding)
                                preprocessing_success_count["npy"] += 1
                            except Exception as e:
                                err_msg = f"Failed to save NPY file {npy_file_path}: {e}"
                                print(f"        ERROR: {err_msg}")
                                preprocessing_errors.append(f"{filename}: {err_msg}")
                                if os.path.exists(npy_file_path):
                                    os.remove(npy_file_path)  # Очистка
                        else:
                            err_msg = f"Failed to generate embedding for {filename}"
                            preprocessing_errors.append(f"{filename}: {err_msg}")
                    elif not extracted_text:
                        pass  # Пропускаем генерацию NPY
                    elif not semantic_model:  # Не должно происходить, так как проверка выполнена в начале
                        err_msg = f"Cannot generate NPY for {filename}, semantic model not loaded."
                        print(f"        ERROR: {err_msg}")
                        preprocessing_errors.append(f"{filename}: {err_msg}")

    except Exception as scan_e:
        print(f"  !!! ERROR during preprocessing scan: {scan_e}")
        preprocessing_errors.append(f"General Scan Error: {scan_e}")

    print(f"  Preprocessing finished. Processed {source_files_processed} source files.")
    print(f"  Created: {preprocessing_success_count['txt']} TXT, {preprocessing_success_count['npy']} NPY.")
    if preprocessing_errors:
        print(f"  Preprocessing Errors ({len(preprocessing_errors)}):")
        for err in preprocessing_errors[:5]:
            print(f"    - {err}")
        if len(preprocessing_errors) > 5:
            print(f"    ... and {len(preprocessing_errors)-5} more.")

    # === Шаг 2: Перезагрузка данных RAG ===
    print(f"\n  Now reloading all data from TXT and EMBEDDING directories...")
    new_chunk_files = []
    new_chunk_texts = []
    new_chunk_tokens = []
    new_chunk_embeddings = []
    new_bm25_index = None

    try:
        # 2a. Сканирование TXT директории
        potential_files = glob.glob(os.path.join(config.TXT_DIR, "*", "*.txt"), recursive=False)
        scanned_files = sorted([f for f in potential_files if os.path.isfile(f)])
        if not scanned_files:
            print("    WARNING: No .txt files found after preprocessing.")
        else:
            print(f"    Found {len(scanned_files)} RAG text files to load.")

        # 2b. Загрузка текстового содержимого
        temp_texts = []
        for fp in scanned_files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    temp_texts.append(f.read())
            except Exception as read_e:
                print(f"      ERROR reading {fp}: {read_e}. Skipping.")
                temp_texts.append(None)
        valid_indices = [i for i, txt in enumerate(temp_texts) if txt is not None]
        if len(valid_indices) != len(scanned_files):
            print(f"    WARNING: {len(scanned_files) - len(valid_indices)} files failed to load/read during final load.")
        new_chunk_files = [scanned_files[i] for i in valid_indices]
        new_chunk_texts = [temp_texts[i] for i in valid_indices]

        # 2c. Токенизация текста
        if new_chunk_texts:
            new_chunk_tokens = [tokenize(text) for text in new_chunk_texts]
        else:
            new_chunk_tokens = []

        # 2d. Загрузка эмбеддингов
        if new_chunk_files:
            loaded_e, error_e, missing_e = 0, 0, 0
            for txt_fp in new_chunk_files:
                embedding = None
                emb_file = None
                try:
                    rel_path = os.path.relpath(txt_fp, start=config.TXT_DIR)
                    base_rel, _ = os.path.splitext(rel_path)
                    emb_file = os.path.join(config.EMBEDDING_DIR, base_rel + '.npy')
                    if os.path.exists(emb_file):
                        try:
                            embedding = np.load(emb_file)
                            loaded_e += 1
                        except Exception as load_e:
                            print(f"      ERROR loading {emb_file}: {load_e}")
                            error_e += 1
                    else:
                        missing_e += 1
                except Exception as path_e:
                    print(f"      ERROR processing path for {txt_fp}: {path_e}")
                    error_e += 1
                new_chunk_embeddings.append(embedding)  # Добавляет None, если отсутствует или произошла ошибка
            print(f"    Embeddings - Loaded: {loaded_e}, Missing: {missing_e}, Errors: {error_e}")
            if len(new_chunk_embeddings) != len(new_chunk_files):
                print("!!! CRITICAL Length mismatch Files/Embeddings post-load!")
        else:
            new_chunk_embeddings = []

        # 2e. Построение индекса BM25
        new_bm25_index = build_bm25_index(new_chunk_tokens)  # Обрабатывает пустой список токенов

        # --- Обновление глобальных переменных ---
        CHUNK_FILES = new_chunk_files
        CHUNK_TEXTS = new_chunk_texts
        CHUNK_TOKENS = new_chunk_tokens
        CHUNK_EMBEDDINGS = new_chunk_embeddings
        BM25_INDEX = new_bm25_index

        total_time = time.time() - start_time
        final_message = (
            f"Preprocessing & Re-indexing complete. "
            f"Created: {preprocessing_success_count['txt']} TXT, {preprocessing_success_count['npy']} NPY. "
            f"Loaded: {len(CHUNK_FILES)} documents. "
            f"Time: {total_time:.2f}s."
        )
        if preprocessing_errors:
            final_message += f" Preprocessing Errors: {len(preprocessing_errors)}."

        print(f"  {final_message}")
        print("--- RAG Re-indexing Process Finished ---")
        return True, final_message

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        message = f"Reloading data failed after preprocessing: {type(e).__name__}: {e}"
        print(f"  !!! ERROR during final data reload:\n{tb_str}")
        print("--- RAG Re-indexing Process Failed During Reload ---")
        return False, message + (f" Preprocessing errors: {len(preprocessing_errors)}." if preprocessing_errors else "")

# === Функции поиска ===
def run_bm25_search(keywords):
    """Выполняет поиск с использованием BM25, фильтрует по порогу и ограничивает количество результатов."""
    global BM25_INDEX, CHUNK_FILES, CHUNK_TEXTS, CHUNK_TOKENS
    if BM25_INDEX is None:
        print("  BM25 search skipped: Index not initialized.")
        return []
    if not CHUNK_FILES or not CHUNK_TEXTS or not CHUNK_TOKENS:
        print("  BM25 search skipped: RAG data lists empty.")
        return []
    if len(CHUNK_FILES) != len(CHUNK_TEXTS) or len(CHUNK_FILES) != len(CHUNK_TOKENS):
        print(f"  ERROR: Mismatch RAG lists! BM25 aborted.")
        return []
    if not isinstance(keywords, list) or not keywords:
        print("  BM25 search skipped: No valid keywords.")
        return []

    query_tokens = tokenize(" ".join(keywords))
    if not query_tokens:
        print("  BM25 search skipped: No tokens from keywords.")
        return []

    try:
        scores = BM25_INDEX.get_scores(query_tokens)
        num_docs = len(CHUNK_FILES)
        # Рассматриваем все документы, сортировка по баллам
        sorted_indices = np.argsort(scores)[::-1]

        threshold_filtered_results = []
        for i in sorted_indices:
            if 0 <= i < num_docs:
                score = scores[i]
                if score > config.BM25_SCORE_THRESHOLD:
                    threshold_filtered_results.append((CHUNK_FILES[i], CHUNK_TEXTS[i], score))
                else:
                    break  # Баллы отсортированы, дальнейшая проверка не требуется
            else:
                print(f"    Warning: BM25 generated invalid index {i}, skipping.")

        final_results = threshold_filtered_results[:config.MAX_CHUNKS_PER_TYPE]
        print(f"  BM25 returning final {len(final_results)} results (Max {config.MAX_CHUNKS_PER_TYPE}).")
        return final_results

    except IndexError as ie:
        # BM25 вызывает IndexError, если токены запроса отсутствуют в словаре индекса
        print(f"  BM25 Search Warning: Query terms likely not found in index vocabulary ({ie}). Keywords: {keywords}")
        return []
    except Exception as e:
        print(f"  Error during BM25 search execution: {e}")
        return []

def run_semantic_search(semantic_query):
    """Выполняет семантический поиск, фильтрует по порогу и ограничивает количество результатов."""
    global semantic_model, CHUNK_FILES, CHUNK_TEXTS, CHUNK_EMBEDDINGS
    if semantic_model is None:
        print("  Semantic search skipped: Model not loaded.")
        return []
    if not CHUNK_FILES or not CHUNK_TEXTS or not CHUNK_EMBEDDINGS:
        print("  Semantic search skipped: RAG lists empty.")
        return []
    if not any(e is not None for e in CHUNK_EMBEDDINGS):
        print("  Semantic search skipped: No valid embeddings.")
        return []
    if len(CHUNK_FILES) != len(CHUNK_TEXTS) or len(CHUNK_FILES) != len(CHUNK_EMBEDDINGS):
        print(f"  ERROR: Mismatch RAG lists! Semantic aborted.")
        return []
    if not isinstance(semantic_query, str) or not semantic_query.strip():
        print("  Semantic search skipped: Invalid query.")
        return []

    try:
        query_embedding = generate_embedding(semantic_query)
        if query_embedding is None:
            print("  Semantic search failed: No query embedding.")
            return []

        similarities = []
        valid_indices = []  # Отслеживаем индексы с корректными эмбеддингами
        for i, doc_embedding in enumerate(CHUNK_EMBEDDINGS):
            if doc_embedding is not None:
                try:
                    sim = cosine_similarity(query_embedding, doc_embedding)
                    similarities.append((sim, i))  # Сохраняем балл и исходный индекс
                except Exception as sim_e:
                    print(f"    Error calc sim idx {i}: {sim_e}")

        # Сортируем по убыванию баллов схожести
        similarities.sort(key=lambda x: x[0], reverse=True)

        threshold_filtered_results = []
        for score, index in similarities:
            if score > config.SEMANTIC_SIMILARITY_THRESHOLD:
                if 0 <= index < len(CHUNK_FILES):
                    threshold_filtered_results.append((CHUNK_FILES[index], CHUNK_TEXTS[index], score))
                    if len(threshold_filtered_results) >= config.MAX_CHUNKS_PER_TYPE:
                        break  # Достигнут лимит
                else:
                    print(f"    Warning: Semantic search index {index} invalid, skipping.")
            else:
                break  # Баллы отсортированы, дальнейшая проверка не требуется

        print(f"  Semantic returning final {len(threshold_filtered_results)} results (Max {config.MAX_CHUNKS_PER_TYPE}).")
        return threshold_filtered_results

    except Exception as e:
        print(f"  Error during semantic search execution: {e}")
        import traceback
        traceback.print_exc()
        return []

# === Объединение результатов ===
def merge_results(bm25_results, semantic_results):
    """Объединяет и удаляет дубликаты результатов, полученных из BM25 и семантического поиска."""
    merged = {}  # Используем путь к файлу в качестве ключа

    # Сначала обрабатываем семантические результаты
    for filepath, text, score in semantic_results:
        if filepath not in merged:
            merged[filepath] = (text, score, "semantic")

    # Обрабатываем результаты BM25
    for filepath, text, score in bm25_results:
        if filepath not in merged:
            merged[filepath] = (text, score, "bm25")
        # Необязательно: можно решить, должна ли более высокая оценка BM25 перезаписывать значение.
        # Текущая логика отдает приоритет семантическим результатам, если путь уже существует.

    final_merged_list = []
    for filepath, (text, score, origin) in merged.items():
        final_merged_list.append((filepath, text, score, origin))  # Сохраняем тег источника

    print(f"  Merged results yield {len(final_merged_list)} unique chunks for LLM.")
    return final_merged_list

# === Вспомогательная функция инициализации (вызывается из app.py) ===
def initialize_rag_resources():
    """
    Загружает модели и инициирует первоначальное построение/загрузку индекса RAG.
    Возвращает True, если успешно, иначе False.
    """
    global semantic_model  # Модель нужно загружать только один раз

    print("\n--- Initializing RAG Resources (Server Startup) ---")

    # Убедиться, что директории существуют
    os.makedirs(config.LESSONS_DIR, exist_ok=True)
    os.makedirs(config.TXT_DIR, exist_ok=True)
    os.makedirs(config.EMBEDDING_DIR, exist_ok=True)
    print(f"  Ensured directories exist: Lessons, Txt, Embeddings")

    # Загрузка семантической модели
    if semantic_model is None:
        try:
            print(f"  Loading semantic model: {config.EMBEDDING_MODEL_NAME}...")
            semantic_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
            print("  Semantic model loaded.")
        except Exception as e:
            print(f"!!! FATAL ERROR loading semantic model ({config.EMBEDDING_MODEL_NAME}): {e} !!!")
            return False  # Критическая ошибка
    else:
        print("  Semantic model already loaded.")

    # Запускаем полную предварительную обработку и построение индекса
    print("  Performing initial RAG data processing and indexing...")
    # Передайте semantic_model явно, если необходимо, или используйте глобальную переменную
    success, message = rebuild_rag_index()  # Вызов объединенной функции
    if not success:
        print(f"!!! WARNING: Initial RAG data load failed: {message} !!!")
        return False  # Считать, что ошибка критическая для запуска
    else:
        print("  Initial RAG data load complete.")

    print("-" * 40)
    return True
