# app.py
import os
import uuid
import time
import threading
import json
import re
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import traceback

# Импорты Flask и связанных библиотек
from flask import Flask, request, jsonify, render_template # send_from_directory удален, так как не используется

# Импорты локальных модулей
import config
import utils # Предполагается, что utils содержит tokenize, truncate_text, isADSIdentifier, sanitize_filename_ru, allowed_file
import rag_core # Предполагается, что rag_core определяет/импортирует run_bm25_search, run_semantic_search, merge_results, initialize_rag_resources, rebuild_rag_index, TXT_DIR, LESSONS_DIR, EMBEDDING_DIR
import llm_interface # Предполагается, что llm_interface определяет/импортирует filter_query_with_llm_v2, generate_final_answer

# --- УСТАНОВКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ ДЛЯ ПРОКСИ (если предоставлены) ---
# Делаем это *перед* инициализацией любых библиотек, которые делают HTTP-запросы (например, genai в llm_interface)
if config.PROXY_URL:
    print(f"--- Configuring proxy for HTTP/HTTPS requests: {config.PROXY_URL} ---")
    os.environ['HTTP_PROXY'] = config.PROXY_URL
    os.environ['HTTPS_PROXY'] = config.PROXY_URL
else:
    print("--- No proxy configured (PROXY_URL is not set in config) ---")

# --- Инициализация приложения Flask ---
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['API_PREFIX'] = config.API_PREFIX
app.config['PROPAGATE_EXCEPTIONS'] = True # Полезно для отладки

# --- Глобальные ресурсы для фоновых задач и кеширования ---
pending_requests = {}               # Хранит статус и результаты асинхронных запросов
pending_requests_lock = threading.Lock() # Блокировка для словаря/очереди pending_requests
rag_resource_lock = threading.Lock() # Блокировка для доступа к данным RAG во время фоновой обработки
request_cache_order = deque()       # Отслеживает порядок запросов для вытеснения из кеша
background_executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

# Флаг, указывающий, готовы ли ресурсы RAG
RAG_RESOURCES_READY = False

# В app.py

def process_request_background(request_id, user_question, filtered_data):
    """
    Полный конвейер RAG, использующий глобальные списки, освобождающий блокировку перед финальным вызовом LLM.
    Включает подробное логирование и ТЕСТ для упрощения структуры полезной нагрузки кеша.
    """
    # --- ОБЪЯВЛЕНИЕ GLOBAL ДОЛЖНО БЫТЬ ПЕРВЫМ ---
    # Эти глобальные переменные изменяются в области видимости этой функции (в частности, pending_requests)
    # или используются в секциях, потенциально защищенных блокировкой (rag_resource_lock).
    global pending_requests, pending_requests_lock, rag_resource_lock
    # -----------------------------------------

    # --- Инициализация локальных переменных ПОСЛЕ global ---
    print(f"[{request_id}] ---> Background RAG Task Started for: '{user_question[:60]}...'")
    start_time = time.time()
    final_status = 'error' # Статус по умолчанию
    # Упрощенная структура полезной нагрузки для тестирования (по умолчанию в случае ранней ошибки):
    result_payload_for_cache = {"final_answer": "Unexpected error at start of background task."}
    context_for_llm = ""
    final_answer = "Error: Processing did not complete." # Ответ по умолчанию
    source_chunks_details = [] # Инициализация списка
    # --- Конец инициализации переменных ---

    try:
        print(f"[{request_id}] Attempting to acquire RAG lock...") # Лог попытки захвата блокировки
        with rag_resource_lock:
            print(f"[{request_id}] Acquired RAG lock.") # Лог захвата блокировки
            # --- Извлечение и слияние под блокировкой ---
            print(f"[{request_id}] Running BM25 Search...") # Лог шага
            # Предполагается, что run_bm25_search корректно импортирован или определен в rag_core
            bm25_results = rag_core.run_bm25_search(filtered_data.get("bm25_keywords", []))
            print(f"[{request_id}] BM25 Search done ({len(bm25_results)} results).") # Лог завершения шага

            print(f"[{request_id}] Running Semantic Search...") # Лог шага
            # Предполагается, что run_semantic_search корректно импортирован или определен в rag_core
            semantic_results = rag_core.run_semantic_search(filtered_data.get("semantic_query", ""))
            print(f"[{request_id}] Semantic Search done ({len(semantic_results)} results).") # Лог завершения шага

            print(f"[{request_id}] Merging Results...") # Лог шага
            # Предполагается, что merge_results корректно импортирован или определен в rag_core
            merged_chunks = rag_core.merge_results(bm25_results, semantic_results)
            print(f"[{request_id}] Merging done ({len(merged_chunks)} unique chunks).") # Лог завершения шага

            # --- Подготовка строки контекста И сбор деталей чанков под блокировкой ---
            print(f"[{request_id}] Preparing context for LLM...") # Лог шага
            if not merged_chunks:
                print(f"  [{request_id}] No relevant chunks found after merge/filter.")
                context_for_llm = ""
                source_chunks_details = [] # Убеждаемся, что он пуст
            else:
                combined_chunks_text = []
                for i, chunk_data in enumerate(merged_chunks):
                    if not isinstance(chunk_data, (list, tuple)) or len(chunk_data) < 3: continue
                    filepath, text, score = chunk_data[0], chunk_data[1], chunk_data[2]
                    origin = f"{chunk_data[3]}" if len(chunk_data) > 3 else "unknown"
                    filename = os.path.basename(filepath)
                    relative_filepath = "N/A"
                    try:
                         # Используем модуль config для TXT_DIR
                         relative_filepath = os.path.relpath(filepath, start=config.TXT_DIR).replace("\\", "/")
                    except ValueError: pass # Если relpath не сработает

                    serializable_score = float(score) if isinstance(score, (np.float32, np.float64)) else score
                    source_chunks_details.append({
                        "filepath": relative_filepath,
                        "filename": filename,
                        "score": serializable_score,
                        "origin": origin
                    })

                    score_info = f"(Score: {serializable_score:.4f} Origin: {origin})" if isinstance(serializable_score, float) else f"(Origin: {origin})"
                    chunk_header = f"**[{filename}] {score_info}:**"
                    # Предполагается, что truncate_text импортирован из utils
                    truncated_content = utils.truncate_text(text, 750)
                    combined_chunks_text.append(f"{chunk_header}\n{truncated_content}")

                context_for_llm = "\n\n---\n\n".join(combined_chunks_text)
            print(f"[{request_id}] Context prepared (length: {len(context_for_llm)}). Releasing RAG lock.") # Лог завершения подготовки контекста

        # --- Блокировка ресурса RAG освобождается здесь ---
        print(f"[{request_id}] Released RAG resource lock.") # Лог освобождения блокировки

        # --- Генерация финального ответа (сетевой вызов) - ВНЕ БЛОКИРОВКИ ---
        print(f"[{request_id}] Generating Final Answer...") # Лог шага
        # Предполагается, что generate_final_answer импортирован из llm_interface
        final_answer = llm_interface.generate_final_answer(user_question, context_for_llm)
        print(f"[{request_id}] Final Answer generation done.") # Лог завершения шага

        # Если до этого момента все успешно, устанавливаем статус и готовим упрощенную полезную нагрузку для теста кеша
        final_status = 'done'
        result_payload_for_cache = {"final_answer": final_answer} # <-- Присваиваем простую полезную нагрузку в случае успеха
        print(f"[{request_id}] RAG Pipeline Completed Successfully. Provided {len(source_chunks_details)} chunks to LLM.")

    except Exception as e:
        # Логируем любое исключение во время основного процесса RAG
        tb_str = traceback.format_exc()
        print(f"[{request_id}] !!! ERROR in background RAG task try block: {type(e).__name__}: {e}\n{tb_str}")
        final_status = 'error'
        error_message = f"Внутренняя ошибка сервера в фоновой задаче ({type(e).__name__})."
        # Устанавливаем final_answer в сообщение об ошибке для упрощенной полезной нагрузки кеша
        result_payload_for_cache = {"final_answer": error_message} # <-- Присваиваем простую полезную нагрузку для ошибки

    # --- Логируем содержимое полезной нагрузки ПЕРЕД входом в блокировку блока finally ---
    print(f"[{request_id}] Payload BEFORE finally lock: Status='{final_status}', Type='{type(result_payload_for_cache)}'")
    try:
        answer_snippet = str(result_payload_for_cache.get('final_answer', 'N/A'))[:100]
        print(f"[{request_id}] Payload Answer Snippet: {answer_snippet}...")
    except Exception as print_e:
         print(f"[{request_id}] Error printing payload snippet before finally: {print_e}")
    # --- Конец логирования полезной нагрузки ---

    finally:
        # --- Обновление кеша ---
        print(f"[{request_id}] FINALLY block: Attempting to acquire cache lock...") # Лог попытки захвата блокировки
        with pending_requests_lock:
            print(f"[{request_id}] FINALLY block: Cache lock acquired.") # Лог захвата блокировки
            if request_id in pending_requests:
                print(f"[{request_id}] FINALLY block: Updating status to '{final_status}'...") # Лог начала обновления статуса
                pending_requests[request_id]['status'] = final_status
                print(f"[{request_id}] FINALLY block: Status updated.") # Лог завершения обновления статуса

                # --- Добавляем ВНУТРЕННИЙ TRY-EXCEPT вокруг обновления данных ---
                try:
                    print(f"[{request_id}] FINALLY block: Attempting to update data payload (SIMPLE TEST)...") # Заметка о тесте
                    # --- ИСПОЛЬЗУЕМ УПРОЩЕННУЮ ПОЛЕЗНУЮ НАГРУЗКУ ДЛЯ ЭТОГО ТЕСТА ---
                    pending_requests[request_id]['data'] = result_payload_for_cache
                    # -------------------------------------------------
                    print(f"[{request_id}] FINALLY block: Data payload update successful.") # Лог завершения обновления данных
                except Exception as payload_update_error:
                    print(f"[{request_id}] !!! CRITICAL ERROR updating data payload INSIDE lock: {payload_update_error} !!!") # Лог конкретной ошибки
                    # Помечаем как ошибку, сохраняем сообщение об ошибке (используя простой формат)
                    pending_requests[request_id]['status'] = 'error'
                    pending_requests[request_id]['data'] = {"final_answer": f"Cache payload update error: {payload_update_error}"}
                # --- КОНЕЦ ВНУТРЕННЕГО TRY-EXCEPT ---

                print(f"[{request_id}] FINALLY block: Updating timestamp...") # Лог начала обновления временной метки
                pending_requests[request_id]['timestamp'] = time.time()
                print(f"[{request_id}] FINALLY block: Timestamp updated.") # Лог завершения обновления временной метки
            else:
                print(f"[{request_id}] !!! WARNING: Request ID not found in cache upon completion (FINALLY block).")

            print(f"[{request_id}] FINALLY block: Releasing cache lock.") # Лог попытки освобождения блокировки
        # --- Блокировка освобождается при выходе из блока 'with' ---
        print(f"[{request_id}] FINALLY block: Cache lock released.") # Лог подтверждения освобождения блокировки

        total_time = time.time() - start_time
        print(f"[{request_id}] <--- Background RAG Task Function Finished (End of Finally). Status: {final_status}. Time: {total_time:.2f} seconds.") # Лог конца функции


# === Управление кешем ===
def cleanup_expired_requests():
    """ Удаляет старые запросы со статусом 'done' или 'error' из кеша на основе TTL и лимита размера. """
    # --- ОБЪЯВЛЕНИЕ GLOBAL ПЕРВЫМ ---
    global pending_requests, request_cache_order, pending_requests_lock
    # --------------------------------

    print("[Cleanup] Starting cleanup...")
    now = time.time()
    ids_to_remove_ttl = []
    ids_to_remove_inconsistent = []

    # Создаем временный список ID для проверки, чтобы не удерживать блокировку дольше необходимого
    # Чтение request_cache_order без блокировки в целом безопасно, если добавления происходят в другом месте
    ids_to_check = list(request_cache_order)

    for req_id in ids_to_check:
         # Чтение из pending_requests без блокировки обычно допустимо для кратковременных проверок
         entry = pending_requests.get(req_id)
         if entry:
             if entry.get('status') in ['done', 'error'] and (now - entry.get('timestamp', 0) > config.REQUEST_CACHE_TTL):
                 ids_to_remove_ttl.append(req_id)
         elif req_id in request_cache_order: # Если он есть в очереди, но не в словаре -> несоответствие
             ids_to_remove_inconsistent.append(req_id)

    print(f"[Cleanup] Found {len(ids_to_remove_ttl)} TTL, {len(ids_to_remove_inconsistent)} inconsistent.")

    # Вытеснение на основе MAX_CACHE_SIZE
    ids_to_remove_evict = []
    # --- Захватываем блокировку ТОЛЬКО для чтения размера и потенциального изменения ---
    print("[Cleanup] Acquiring lock for eviction check/modify...")
    with pending_requests_lock:
        print("[Cleanup] Lock acquired.")
        current_cache_size = len(pending_requests)
        # Вычисляем, сколько ДОПОЛНИТЕЛЬНО удалить ПОСЛЕ удаления по TTL
        num_to_evict = max(0, (current_cache_size - len(ids_to_remove_ttl) - len(ids_to_remove_inconsistent)) - config.MAX_CACHE_SIZE)
        print(f"[Cleanup] Cache size: {current_cache_size}, Need to evict (for size): {num_to_evict}")

        if num_to_evict > 0:
            # Находим самые старые элементы в очереди, которые еще не помечены для удаления
            candidates_for_eviction = [
                req_id for req_id in request_cache_order
                if req_id not in ids_to_remove_ttl and req_id not in ids_to_remove_inconsistent
            ]
            ids_to_remove_evict = candidates_for_eviction[:num_to_evict]
            print(f"[Cleanup] Evicting oldest {len(ids_to_remove_evict)} for size.")

        all_ids_to_remove = set(ids_to_remove_ttl + ids_to_remove_inconsistent + ids_to_remove_evict)
        print(f"[Cleanup] Total entries to remove: {len(all_ids_to_remove)}")

        if all_ids_to_remove:
            cleaned_count = 0
            new_cache_order = deque() # Создаем новую очередь вместо удаления на месте
            # Удаляем из словаря
            for req_id in all_ids_to_remove:
                if req_id in pending_requests:
                    del pending_requests[req_id]
                    cleaned_count += 1
            # Перестраиваем очередь из исходной, пропуская удаленные ID
            for req_id in request_cache_order:
                 if req_id not in all_ids_to_remove:
                      new_cache_order.append(req_id)
            request_cache_order = new_cache_order # Присваиваем новую очередь

            print(f"[Cleanup] Removed {cleaned_count} entries. New size: {len(pending_requests)}")
        print("[Cleanup] Releasing lock.")
    print("[Cleanup] Lock released. Cleanup finished.")


# === Маршруты Flask ===

@app.route(f"{config.API_PREFIX}/ask", methods=["POST"])
def ask_v2_root_handler():
    """ Инициирует асинхронный RAG-запрос. """
    # --- ОБЪЯВЛЕНИЕ GLOBAL ПЕРВЫМ ---
    # Необходимо, так как RAG_RESOURCES_READY читается/проверяется
    global RAG_RESOURCES_READY
    # pending_requests, pending_requests_lock, request_cache_order изменяются
    # background_executor используется
    global pending_requests, pending_requests_lock, request_cache_order, background_executor
    # --------------------------------

    print(f"\n--- Request Received: POST {request.path} ---")
    if not RAG_RESOURCES_READY:
        print("[Ask Error] RAG system resources not ready.")
        return jsonify({"error": "Ошибка сервера: ресурсы системы RAG инициализируются или неисправны. Пожалуйста, повторите попытку позже."}), 503

    if not request.is_json:
        return jsonify({"error": "Запрос должен быть в формате JSON"}), 415

    try:
        data = request.get_json()
        if data is None: raise ValueError("Не получены данные JSON")
        user_question = data.get("question", "").strip()
        if not user_question: raise ValueError("Поле 'question' обязательно")
    except Exception as e:
        print(f"[Ask Error] Invalid JSON or missing/empty 'question': {e}")
        return jsonify({"error": f"Неверный формат JSON или отсутствует/пустое поле 'question': {e}"}), 400

    request_id = str(uuid.uuid4())
    print(f"[Req {request_id}] Processing question: '{user_question[:70]}...'")

    # Фильтруем запрос с помощью LLM (теперь в llm_interface)
    try:
        # Предполагается, что filter_query_with_llm_v2 импортирован из llm_interface
        filtered = llm_interface.filter_query_with_llm_v2(user_question)
        print(f"[Req {request_id}] LLM filtering successful.") # Логируем успех *перед* возможными зависаниями
    except Exception as filter_e:
        print(f"[Req {request_id}] CRITICAL Error during initial LLM filter call: {filter_e}. Using fallback.")
        # Убеждаемся, что структура запасного варианта соответствует ожидаемой структуре 'filtered'
        filtered = {
            "confirmation": f"Ошибка системы обработки ИИ ({type(filter_e).__name__}). Ищу по тексту: '{user_question[:50]}...'",
            "semantic_query": user_question,
            # Предполагается, что tokenize импортирован из utils
            "bm25_keywords": utils.tokenize(user_question)[:5]
        }

    # --- Запускаем очистку *ПЕРЕД* захватом блокировки для добавления нового запроса ---
    try:
        print(f"[Req {request_id}] Running cache cleanup...")
        cleanup_start_time = time.time()
        cleanup_expired_requests()
        cleanup_duration = time.time() - cleanup_start_time
        print(f"[Req {request_id}] Cache cleanup finished (Duration: {cleanup_duration:.4f}s).")
    except Exception as cleanup_e:
        print(f"[Req {request_id}] !!! ERROR during cache cleanup: {cleanup_e} !!!")
        # Продолжаем обработку, но логируем ошибку

    # --- Захватываем блокировку ТОЛЬКО для быстрой операции добавления ---
    print(f"[Req {request_id}] Acquiring cache lock to add request...")
    with pending_requests_lock:
        print(f"[Req {request_id}] Cache lock acquired.")
        pending_requests[request_id] = {
            'status': 'processing',
            'data': None, # Будет заполнено фоновой задачей
            'timestamp': time.time()
        }
        request_cache_order.append(request_id)
        print(f"[Req {request_id}] Added to cache (Size: {len(pending_requests)}). Releasing lock.")
    print(f"[Req {request_id}] Cache lock released.")

    # Отправляем обработку RAG в фоновый пул
    try:
        print(f"[Req {request_id}] Submitting RAG task to background executor...")
        background_executor.submit(process_request_background, request_id, user_question, filtered)
        print(f"[Req {request_id}] Background task submitted successfully.")
    except Exception as e:
        print(f"[Req {request_id}] !!! CRITICAL ERROR submitting background task: {e} !!!")
        # Пытаемся откатить запись в кеше, если отправка не удалась?
        with pending_requests_lock:
            if request_id in pending_requests:
                pending_requests[request_id]['status'] = 'error'
                pending_requests[request_id]['data'] = {"final_answer": "Failed to initiate background processing."} # Упрощенная полезная нагрузка для ошибки
                pending_requests[request_id]['timestamp'] = time.time()
        return jsonify({"error": "Не удалось начать обработку запроса на сервере."}), 500

    # Возвращаем первоначальный ответ
    print(f"[Req {request_id}] Preparing initial response...")
    confirmation_text = filtered.get("confirmation", "Ваш запрос принят в обработку...")
    initial_response = {
        "request_id": request_id,
        "confirmation": confirmation_text,
        "status": "processing"
    }
    print(f"[Req {request_id}] Sending initial response (202 Accepted)")
    return jsonify(initial_response), 202


@app.route(f"{config.API_PREFIX}/get_answer", methods=["GET"])
def get_answer_v2():
    """
    Опрашивает статус/результат ранее отправленного запроса.
    Адаптируется к упрощенной полезной нагрузке кеша, используемой в тесте.
    """
    # --- Глобальное объявление ---
    # Необходимо для чтения pending_requests и использования его блокировки
    global pending_requests, pending_requests_lock
    # --------------------------

    request_id = request.args.get("request_id")
    if not request_id:
        return jsonify({"error": "Параметр request_id обязателен"}), 400

    with pending_requests_lock: # Блокировка необходима для безопасного чтения из словаря
        # Делаем копию, чтобы быстрее освободить блокировку
        request_data = pending_requests.get(request_id)
        if request_data:
            request_data = request_data.copy()

    if request_data:
        status = request_data.get('status', 'unknown')
        data = request_data.get('data') # Это поле 'data' теперь содержит упрощенные данные {"final_answer": ...}
        response_data = {"request_id": request_id, "status": status}

        if status == 'done':
            # --- Адаптация к упрощенной полезной нагрузке ---
            result_payload = data if isinstance(data, dict) else {}
            response_data["result"] = {
                "final_answer": result_payload.get("final_answer", "Ошибка: Финальный ответ отсутствует."),
                "source_chunks": [] # В этой тестовой версии чанки не сохраняются
            }
            # -----------------------------------
            print(f"[Req {request_id}] Status is 'done'. Sending final answer (NO chunks). 200 OK.")
            return jsonify(response_data), 200
        elif status == 'error':
            # --- Адаптация к упрощенной полезной нагрузке ---
            error_message = "Произошла неизвестная ошибка."
            if isinstance(data, dict):
                error_message = data.get("final_answer", error_message)
            elif isinstance(data, str):
                 error_message = data
            # -----------------------------------
            response_data["message"] = error_message
            print(f"[Req {request_id}] Status is 'error'. Sending 500.")
            return jsonify(response_data), 500
        elif status == 'processing':
            response_data["message"] = "Результат еще обрабатывается."
            return jsonify(response_data), 202
        else:
            response_data["message"] = f"Запрос найден, но имеет неожиданный статус: {status}"
            return jsonify(response_data), 500
    else:
        return jsonify({
            "request_id": request_id, "status": "not_found",
            "error": "Идентификатор запроса не найден. Он может быть недействительным, истекшим или задача еще не началась/не зарегистрирована."
        }), 404


@app.route(f"{config.API_PREFIX}/materials", methods=["GET"])
def materials_v2():
    """ Перечисляет уроки и их файлы, указывая статус индексации. """
    print(f"\n--- Request Received: GET {request.path} ---")
    # Предполагается, что config определяет LESSONS_DIR и EMBEDDING_DIR
    lessons_base_dir = config.LESSONS_DIR
    embeddings_base_dir = config.EMBEDDING_DIR

    if not os.path.isdir(lessons_base_dir):
        print(f"[Materials Error] Lessons base directory not found: {lessons_base_dir}")
        return jsonify({"error": f"Базовый каталог '{os.path.basename(lessons_base_dir)}' не найден."}), 404

    lessons_output = {}
    try:
        lesson_folders = [d for d in os.listdir(lessons_base_dir) if os.path.isdir(os.path.join(lessons_base_dir, d))]

        for lesson_folder_name in sorted(lesson_folders):
            lesson_path = os.path.join(lessons_base_dir, lesson_folder_name)
            lesson_files_details = []
            try:
                items = os.listdir(lesson_path)
                # Предполагается, что isADSIdentifier импортирован из utils
                files_in_lesson = sorted([
                    f for f in items
                    if os.path.isfile(os.path.join(lesson_path, f)) and not utils.isADSIdentifier(f)
                ], key=str.lower)

                for filename in files_in_lesson:
                    base_filename, _ = os.path.splitext(filename)
                    expected_embedding_path = os.path.join(embeddings_base_dir, lesson_folder_name, base_filename + ".npy")
                    is_indexed = os.path.exists(expected_embedding_path)
                    lesson_files_details.append({"name": filename, "indexed": is_indexed})
                lessons_output[lesson_folder_name] = lesson_files_details

            except OSError as e:
                print(f"Error listing files in lesson '{lesson_path}': {e}")
                lessons_output[lesson_folder_name] = {"error": f"Не удалось прочитать файлы: {e}"}
            except Exception as e:
                 print(f"Unexpected error listing files in '{lesson_path}': {e}")
                 lessons_output[lesson_folder_name] = {"error": f"Неожиданная ошибка: {e}"}

    except OSError as e:
        print(f"Error accessing base lessons directory '{lessons_base_dir}': {e}")
        return jsonify({"error": "Не удалось прочитать базовый каталог уроков."}), 500
    except Exception as e:
         print(f"An unexpected error occurred during material listing: {e}")
         return jsonify({"error": "Произошла неожиданная ошибка сервера."}), 500

    print(f"Found {len(lessons_output)} lesson folders.")
    return jsonify(lessons_output)


@app.route(f"{config.API_PREFIX}/materials/upload", methods=["POST"])
def upload_material():
    """ Обрабатывает загрузку файлов в указанный подкаталог урока. """
    print(f"\n--- Request Received: POST {request.path} ---")
    # Предполагается, что config определяет UPLOAD_FOLDER и ALLOWED_EXTENSIONS
    base_dir = config.UPLOAD_FOLDER

    if 'file' not in request.files:
        return jsonify({"error": "Отсутствует часть 'file'"}), 400

    lesson_name = request.form.get('lesson_name', '').strip()
    if not lesson_name:
        return jsonify({"error": "Не указано 'lesson_name'"}), 400

    safe_lesson_name = re.sub(r'[^\w\-]+', '_', lesson_name).strip('_')[:100]
    if not safe_lesson_name:
         return jsonify({"error": "Недопустимое имя урока 'lesson_name' (пустое после очистки)"}), 400
    print(f"[Upload] Target lesson (sanitized): '{safe_lesson_name}'")

    target_lesson_dir = os.path.join(base_dir, safe_lesson_name)
    abs_target_lesson_dir = os.path.abspath(target_lesson_dir)
    abs_base_dir = os.path.abspath(base_dir)
    if not abs_target_lesson_dir.startswith(abs_base_dir):
        print(f"[SECURITY ALERT] Path traversal attempt: {target_lesson_dir}")
        return jsonify({"error": "Недопустимый путь (риск безопасности)."}), 400

    try:
        os.makedirs(target_lesson_dir, exist_ok=True)
    except Exception as e:
        print(f"[Upload Error] Could not create directory '{target_lesson_dir}': {e}")
        return jsonify({"error": f"Не удалось создать каталог '{safe_lesson_name}'."}), 500

    uploaded_files_info = []
    errors = []
    files = request.files.getlist('file')

    if not files or all(f.filename == '' for f in files):
         return jsonify({"error": "Файлы не выбраны"}), 400

    for file in files:
        if file and file.filename:
             original_filename = file.filename
             print(f"[Upload] Processing file: '{original_filename}'")

             # Предполагается, что allowed_file и sanitize_filename_ru импортированы из utils
             if not utils.allowed_file(original_filename, config.ALLOWED_EXTENSIONS):
                 error_msg = f"Тип файла '{original_filename.rsplit('.', 1)[-1]}' не разрешен."
                 errors.append({"filename": original_filename, "error": error_msg})
                 print(f"[Upload] Rejected (disallowed type): {original_filename}")
                 continue

             filename = utils.sanitize_filename_ru(original_filename)
             target_path = os.path.join(target_lesson_dir, filename)

             try:
                 file.save(target_path)
                 uploaded_files_info.append({
                     "lesson": safe_lesson_name, "filename": filename,
                     "original_filename": original_filename,
                 })
                 print(f"[Upload] Successfully saved: {target_path}")
             except Exception as e:
                 print(f"[Upload Error] Could not save '{filename}': {e}")
                 errors.append({"filename": original_filename, "error": f"Ошибка сервера при сохранении файла: {e}"})
        elif file.filename == '': pass

    if uploaded_files_info:
         print("\n***************************************************************")
         print("**** Загрузка файла успешна! НАПОМИНАНИЕ:              ****")
         print("**** Индекс RAG НЕ обновлен автоматически. ПЕРЕЗАПУСК или ПЕРЕИНДЕКСАЦИЯ! ****")
         print("***************************************************************\n")

    if not uploaded_files_info and errors:
         return jsonify({"message": "Загрузка не удалась для всех файлов.", "errors": errors}), 400
    elif errors:
         return jsonify({
             "message": f"Загрузка в урок '{safe_lesson_name}' завершена с ошибками.",
             "uploaded": uploaded_files_info, "errors": errors
         }), 207
    elif uploaded_files_info:
        return jsonify({
            "message": f"Успешно загружено {len(uploaded_files_info)} файл(а/ов) в урок '{safe_lesson_name}'.",
            "uploaded": uploaded_files_info
        }), 201
    else:
        return jsonify({"error": "Не было обработано ни одного допустимого файла."}), 400


@app.route(f"{config.API_PREFIX}/materials/delete", methods=["DELETE"])
def delete_materials():
    """ Удаляет указанные файлы и пытается удалить связанные .txt/.npy. """
    print(f"\n--- Request Received: DELETE {request.path} ---")
    # Предполагается, что config определяет LESSONS_DIR, TXT_DIR, EMBEDDING_DIR
    lessons_base_dir = config.LESSONS_DIR
    txt_base_dir = config.TXT_DIR
    embeddings_base_dir = config.EMBEDDING_DIR

    if not request.is_json:
        return jsonify({"error": "Запрос должен быть в формате JSON"}), 415

    try:
        data = request.get_json()
        files_to_delete = data.get("files_to_delete")
        if not isinstance(files_to_delete, list) or \
           not all(isinstance(f, str) and f.strip() for f in files_to_delete):
             raise ValueError("'files_to_delete' должен быть непустым списком строк.")
    except Exception as e:
         return jsonify({"error": f"Неверный формат JSON или список 'files_to_delete': {e}"}), 400

    if not files_to_delete:
        return jsonify({"message": "Файлы не указаны.", "deleted_count": 0, "deleted_folders": [], "errors": []}), 200

    deleted_files_count = 0
    deleted_folders_count = 0
    errors = []
    potentially_empty_folders = set()
    deleted_folders_list = []

    print(f"[Delete] Attempting to delete {len(files_to_delete)} items and related files...")

    for item_path_from_request in files_to_delete:
        relative_path = item_path_from_request.strip()
        if ':' in relative_path and "Zone.Identifier" in relative_path:
            relative_path = relative_path.split(':', 1)[0]
        if '' in relative_path and "Zone.Identifier" in relative_path:
             relative_path = relative_path.split('', 1)[0]

        if not relative_path or '..' in relative_path or os.path.isabs(relative_path) or relative_path.startswith('/'):
             errors.append({"file": item_path_from_request, "error": "Недопустимый/небезопасный формат пути"})
             continue

        try:
            lesson_folder_name, filename = os.path.split(relative_path)
            if not lesson_folder_name or not filename: raise ValueError("Путь должен включать папку и файл.")
        except ValueError as e:
             errors.append({"file": item_path_from_request, "error": f"Неверный формат пути ('папка/файл'): {e}"})
             continue

        lesson_file_path = os.path.join(lessons_base_dir, lesson_folder_name, filename)
        base_filename_no_ext, _ = os.path.splitext(filename)
        txt_file_path = os.path.join(txt_base_dir, lesson_folder_name, base_filename_no_ext + ".txt")
        npy_file_path = os.path.join(embeddings_base_dir, lesson_folder_name, base_filename_no_ext + ".npy")
        parent_lesson_folder_path = os.path.dirname(lesson_file_path)

        abs_lesson_file_path = os.path.abspath(lesson_file_path)
        abs_lessons_base_dir = os.path.abspath(lessons_base_dir)
        if not abs_lesson_file_path.startswith(abs_lessons_base_dir):
            errors.append({"file": item_path_from_request, "error": "Нарушение безопасности: Недопустимый целевой путь."})
            continue

        deleted_main = False
        try:
            if os.path.exists(lesson_file_path):
                os.remove(lesson_file_path)
                print(f"  [Delete OK] Deleted lesson file: {lesson_file_path}")
                deleted_main = True
                potentially_empty_folders.add(parent_lesson_folder_path)
            else:
                print(f"  [Delete Warn] Lesson file not found: {lesson_file_path}")
                potentially_empty_folders.add(parent_lesson_folder_path)

            if os.path.exists(txt_file_path):
                 try: os.remove(txt_file_path); print(f"  [Delete OK] Deleted TXT: {txt_file_path}")
                 except OSError as txt_e: errors.append({"file": f"{relative_path} (TXT)", "error": f"Не удалось: {txt_e}"})

            if os.path.exists(npy_file_path):
                 try: os.remove(npy_file_path); print(f"  [Delete OK] Deleted NPY: {npy_file_path}")
                 except OSError as npy_e: errors.append({"file": f"{relative_path} (NPY)", "error": f"Не удалось: {npy_e}"})

            if deleted_main: deleted_files_count += 1

        except OSError as e: errors.append({"file": item_path_from_request, "error": f"Ошибка ОС: {e}"})
        except Exception as e: errors.append({"file": item_path_from_request, "error": f"Неожиданная ошибка: {e}"})

    abs_lessons_base_dir = os.path.abspath(lessons_base_dir)
    if potentially_empty_folders:
        print(f"[Delete] Checking {len(potentially_empty_folders)} folders for potential cleanup...")
        for folder_path in potentially_empty_folders:
            abs_folder_path = os.path.abspath(folder_path)
            if not abs_folder_path.startswith(abs_lessons_base_dir) or abs_folder_path == abs_lessons_base_dir:
                continue
            try:
                if os.path.isdir(folder_path) and not os.listdir(folder_path):
                    os.rmdir(folder_path)
                    deleted_folders_count += 1
                    deleted_folders_list.append(os.path.basename(folder_path))
                    print(f"  [Delete OK] Removed empty lesson folder: {folder_path}")
            except OSError as e: print(f"  [Delete Warn] Could not remove folder {folder_path}: {e}")
            except Exception as e: print(f"  [Delete Error] Unexpected error removing folder {folder_path}: {e}")

    response_status = 200
    if errors and deleted_files_count == 0 and deleted_folders_count == 0: response_status = 400
    elif errors: response_status = 207
    message = f"Удалено файлов: {deleted_files_count}. Удалено пустых папок: {deleted_folders_count}."
    if response_status == 400: message = "Удаление не удалось для всех указанных элементов."
    elif response_status == 207: message = f"Удаление завершено с ошибками. {message}"
    else: message = f"Удаление успешно. {message}"

    print(f"[Delete] Finished. Status: {response_status}, Files: {deleted_files_count}, Folders: {deleted_folders_count}, Errors: {len(errors)}")

    if deleted_files_count > 0 or deleted_folders_count > 0:
         print("\n***************************************************************")
         print("**** Удаление файла успешно! НАПОМИНАНИЕ:                ****")
         print("**** Индекс RAG устарел. ПЕРЕЗАПУСК или ПЕРЕИНДЕКСАЦИЯ!   ****")
         print("***************************************************************\n")

    return jsonify({
        "message": message, "deleted_count": deleted_files_count,
        "deleted_folders": deleted_folders_list, "errors": errors
    }), response_status


@app.route("/upload", methods=["GET"])
def upload_page():
    """ Обслуживает простую HTML-страницу загрузки. """
    print(f"\n--- Request Received: GET {request.path} ---")
    # Предполагается, что config определяет API_PREFIX
    return render_template("upload.html", api_prefix=config.API_PREFIX)


@app.route(f"{config.API_PREFIX}/materials/reindex", methods=["POST"])
def reindex_materials():
    """ Запускает перестроение индекса RAG (требует заголовок аутентификации). """
    # --- Глобальное объявление ---
    # Необходимо для изменения RAG_RESOURCES_READY и использования rag_resource_lock
    global RAG_RESOURCES_READY, rag_resource_lock
    # --------------------------

    print(f"\n--- Request Received: POST {request.path} ---")

    auth_header = request.headers.get("X-Reindex-Secret")
    # Предполагается, что config определяет REINDEX_SECRET
    if not auth_header or auth_header != config.REINDEX_SECRET:
        print("[Reindex Error] Authentication failed.")
        return jsonify({"error": "Не авторизован"}), 401

    print("[Reindex] Authentication successful. Starting re-index...")
    RAG_RESOURCES_READY = False # Помечаем как не готовый во время переиндексации
    try:
        print("[Reindex] Acquiring RAG resource lock for rebuild...")
        with rag_resource_lock:
            print("[Reindex] Acquired lock. Calling rebuild_rag_index...")
            # Предполагается, что rebuild_rag_index импортирован из rag_core
            success, message = rag_core.rebuild_rag_index() # Это блокирующая операция

        print("[Reindex] Released RAG resource lock.")

        if success:
            RAG_RESOURCES_READY = True # Снова помечаем как готовый
            print("[Reindex] Re-index completed successfully.")
            return jsonify({"message": message}), 200
        else:
            print(f"[Reindex Error] Re-indexing process failed: {message}")
            return jsonify({"error": f"Переиндексация не удалась: {message}"}), 500
    except Exception as e:
        RAG_RESOURCES_READY = False # Убеждаемся, что он помечен как не готовый при неожиданной ошибке
        print(f"[Reindex Error] Unexpected error in endpoint: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Ошибка сервера во время запроса на переиндексацию: {e}"}), 500


# === Основное выполнение ===
if __name__ == "__main__":
    print("--- Starting RAG Chatbot Application ---")

    # Инициализируем ресурсы RAG ОДИН РАЗ при запуске
    # Предполагается, что initialize_rag_resources импортирован из rag_core
    RAG_RESOURCES_READY = rag_core.initialize_rag_resources()
    if not RAG_RESOURCES_READY:
        print("!!! CRITICAL: Failed to initialize RAG resources. API may not function correctly. !!!")
        # Рассмотреть выход, если инициализация критична

    print("\n--- Flask Development Server Configuration ---")
    # Предполагается, что необходимые переменные определены в config
    print(f"  - API Prefix:             {config.API_PREFIX}")
    print(f"  - Lessons/Upload Dir:     {config.LESSONS_DIR}")
    print(f"  - RAG Source Text Dir:    {config.TXT_DIR}")
    print(f"  - RAG Embeddings Dir:     {config.EMBEDDING_DIR}")
    print(f"  - RAG Workers:            {config.MAX_WORKERS}")
    # Доступ к CHUNK_FILES требует импорта rag_core или объявления его глобальным в app.py
    # print(f"  - Initial RAG docs found: {len(rag_core.CHUNK_FILES)}") # Требует импорта rag_core и завершения инициализации
    print(f"  - RAG Ready Status:       {RAG_RESOURCES_READY}")
    if config.PROXY_URL: print(f"  - Proxy Configured:       {config.PROXY_URL}")
    else: print(f"  - Proxy Configured:       No")
    print("-" * 40)
    print(f"Access RAG API (POST): http://127.0.0.1:5002{config.API_PREFIX}/ask")
    print(f"Access Upload Page:    http://127.0.0.1:5002/upload")
    print(f"Access Materials List: http://127.0.0.1:5002{config.API_PREFIX}/materials")
    print(f"Trigger Re-index (POST): http://127.0.0.1:5002{config.API_PREFIX}/materials/reindex (Requires X-Reindex-Secret header)")
    print("-" * 40)
    print("Starting Flask server...")

    # Используйте threaded=True для фоновых задач/опроса. Установите debug=False для продакшена.
    app.run(host="127.0.0.1", port=5002, debug=False, threaded=True)

    print("--- Flask Server Stopped ---")
    print("Shutting down background executor...")
    background_executor.shutdown(wait=True) # Ждем, пока незавершенные задачи попытаются выполниться
    print("Executor shut down.")