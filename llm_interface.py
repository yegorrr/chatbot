# llm_interface.py
import json
import google.generativeai as genai
# Импорт Exception DeadlineExceeded для отдельной обработки
from google.api_core import exceptions as google_exceptions

import config
from utils import clean_text, load_prompt_template, tokenize

# --- Загрузка шаблонов подсказок ---
SYSTEM_FILTER_PROMPT = load_prompt_template(config.SYSTEM_FILTER_PROMPT_FILE)
FINAL_ANSWER_PROMPT_TEMPLATE = load_prompt_template(config.FINAL_ANSWER_PROMPT_FILE)

# --- Настройка Google AI ---
try:
    genai.configure(api_key=config.GEMINI_API_KEY)
    print("--- Google AI SDK configured via llm_interface. ---")
except Exception as e:
    print(f"!!! FATAL ERROR configuring Google AI SDK in llm_interface: {e} !!!")

# --- Функции взаимодействия с LLM ---
def filter_query_with_llm_v2(user_question: str) -> dict:
    """
    Использует LLM (Gemini) для преобразования запроса пользователя в структурированные поисковые термины.
    Включает резервную логику и обработку таймаута.
    """
    print(f"  Filtering query with LLM ({config.LLM_MODEL_NAME}): '{user_question[:60]}...'")
    if not config.GEMINI_API_KEY or "YOUR_API_KEY" in config.GEMINI_API_KEY or ("AIzaSy" not in config.GEMINI_API_KEY):
         return { "confirmation": f"Ошибка конфигурации AI. Ищу: '{user_question[:50]}...'", "semantic_query": user_question, "bm25_keywords": tokenize(user_question)[:5] }

    if "ERROR:" in SYSTEM_FILTER_PROMPT:  # Проверка: шаблон системы не загружен
         print(f"  ERROR: Cannot filter query, system prompt failed to load.")
         return { "confirmation": f"Ошибка загрузки системной инструкции AI. Ищу: '{user_question[:50]}...'", "semantic_query": user_question, "bm25_keywords": tokenize(user_question)[:5] }

    model = genai.GenerativeModel(config.LLM_MODEL_NAME)
    full_prompt = clean_text(SYSTEM_FILTER_PROMPT + "\nЗапрос пользователя: " + user_question)
    response = None
    content = ""

    try:
        # --- ДОБАВЛЯЕМ ТАЙМАУТ ЗДЕСЬ ---
        request_options = {"timeout": 30}  # Таймаут в секундах (например, 30 секунд для фильтрации)
        response = model.generate_content(
            full_prompt,
            generation_config=config.generation_config,
            request_options=request_options  # Передаем параметр таймаута
        )
        try:
            content = response.text.strip()
        except ValueError:
            finish_reason = getattr(getattr(response, 'prompt_feedback', None), 'block_reason', 'UNKNOWN')
            print(f"    WARNING: Could not access filter response text. Blocked? Reason: {finish_reason}")
            content = ""
            # Определяем, нужно ли генерировать ошибку для конкретных причин блокировки
            if finish_reason == 'SAFETY':
                raise ValueError(f"Filter response blocked: {finish_reason}")
        except Exception as resp_err:
            print(f"    ERROR accessing filter response text: {resp_err}")
            content = ""
        if not content:
            raise ValueError("LLM filter response was empty.")

        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        try:
            parsed_json = json.loads(content)
        except json.JSONDecodeError as json_e:
            raw_resp_text = getattr(response, 'text', 'Error accessing response.text')
            print(f"    ERROR parsing LLM filter JSON response: {json_e}. Raw: >>>{raw_resp_text}<<<")
            raise ValueError(f"LLM filter returned invalid JSON: {json_e}")

        if not isinstance(parsed_json, dict) or \
           not all(k in parsed_json for k in ["confirmation", "semantic_query", "bm25_keywords"]) or \
           not isinstance(parsed_json.get("semantic_query"), str) or \
           not isinstance(parsed_json.get("bm25_keywords"), list):
             print(f"    ERROR: LLM filter JSON missing keys/wrong types: {parsed_json}")
             raise ValueError("LLM filter JSON structure incorrect.")

        parsed_json["bm25_keywords"] = [str(kw) for kw in parsed_json["bm25_keywords"] if kw]
        print("  LLM filtering successful.")
        return parsed_json

    # --- ОБРАБОТКА ОШИБКИ: таймаут ---
    except google_exceptions.DeadlineExceeded as e:
        print(f"  ERROR: Timeout during filter_query_with_llm_v2: {e}")
        # Резервная логика
        return {
            "confirmation": f"Ошибка сети (таймаут) при обработке ИИ. Ищу по тексту: '{user_question[:50]}...'",
            "semantic_query": user_question,
            "bm25_keywords": tokenize(user_question)[:5]
        }
    except Exception as e:
        print(f"  ERROR during filter_query_with_llm_v2: {type(e).__name__}: {e}")
        return {
            "confirmation": f"Произошла ошибка при обработке запроса ИИ ({type(e).__name__}). Ищу по тексту: '{user_question[:50]}...'",
            "semantic_query": user_question,
            "bm25_keywords": tokenize(user_question)[:5]
        }

def generate_final_answer(user_question, combined_chunks_str):
    """
    Генерирует финальный ответ с использованием LLM. Включает обработку таймаута сети.
    """
    if not combined_chunks_str:
        return ("К сожалению, в материалах курса не нашлось информации по вашему запросу. "
                "Попробуйте переформулировать вопрос.")

    print(f"  Generating final answer based on provided context string (length {len(combined_chunks_str)}).")
    if "ERROR:" in FINAL_ANSWER_PROMPT_TEMPLATE:
        return "Ошибка: Не удалось загрузить шаблон ответа AI."

    try:
        prompt = FINAL_ANSWER_PROMPT_TEMPLATE.format(
            user_question=user_question,
            combined_chunks_str=combined_chunks_str
        )
    except KeyError as e:
         return f"Ошибка: Неверный формат шаблона ответа AI (отсутствует {e})."

    # --- Вызов LLM ---
    try:
        if not config.GEMINI_API_KEY or "YOUR_API_KEY" in config.GEMINI_API_KEY or ("AIzaSy" not in config.GEMINI_API_KEY):
             return "Ошибка: Ключ API для генерации ответа не настроен."

        model = genai.GenerativeModel(config.LLM_MODEL_NAME)
        cleaned_prompt = clean_text(prompt)

        # --- ДОБАВЛЯЕМ ТАЙМАУТ ЗДЕСЬ ---
        request_options = {"timeout": 90}  # Более длительный таймаут для генерации (например, 90 секунд)
        response = model.generate_content(
            cleaned_prompt,
            generation_config=config.generation_config,
            request_options=request_options  # Передача параметра таймаута
        )

        final_text = ""
        try:
            final_text = response.text.strip()
            if not final_text:
                finish_reason = "Unknown"
                try:
                    finish_reason = response.candidates[0].finish_reason.name
                except (AttributeError, IndexError):
                    pass
                print(f"    WARNING: LLM final answer response content is empty. Finish Reason: {finish_reason}")
                if finish_reason == 'SAFETY':
                    final_text = f"Не удалось сгенерировать ответ из-за настроек безопасности контента."
                elif finish_reason == 'RECITATION':
                    final_text = f"Не удалось сгенерировать ответ из-за политики цитирования."
                else:
                    final_text = f"Не удалось сгенерировать ответ (пустой ответ от модели, причина: {finish_reason})."
        except ValueError as ve:  # Проверка блокировки ответа
             block_reason = "Unknown"
             try:
                 block_reason = response.prompt_feedback.block_reason.name
             except AttributeError:
                 pass
             print(f"    WARNING: LLM final answer response blocked or empty. Reason: {block_reason}")
             final_text = f"Не удалось сгенерировать ответ (заблокировано: {block_reason}). Попробуйте переформулировать."
        except Exception as resp_err:
             print(f"    ERROR accessing LLM final answer response content: {resp_err}")
             final_text = "Ошибка получения текста ответа от языковой модели."

        print("  Final answer received from LLM.")
        return final_text

    # --- ОБРАБОТКА ОШИБКИ: таймаут ---
    except google_exceptions.DeadlineExceeded as e:
        print(f"  ERROR: Timeout during final answer generation call to LLM: {e}")
        return "Произошла ошибка: превышено время ожидания ответа от языковой модели при генерации ответа."
    except Exception as e:
        print(f"  ERROR during final answer generation call to LLM: {type(e).__name__}: {e}")
        return f"Произошла внутренняя ошибка ({type(e).__name__}) при генерации финального ответа. Пожалуйста, попробуйте позже."
