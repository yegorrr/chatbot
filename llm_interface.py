# llm_interface.py
import json
import openai
import httpx # Для обработки таймаутов

import config
from utils import clean_text, load_prompt_template, tokenize

# --- Загрузка шаблонов подсказок ---
SYSTEM_FILTER_PROMPT = load_prompt_template(config.SYSTEM_FILTER_PROMPT_FILE)
FINAL_ANSWER_PROMPT_TEMPLATE = load_prompt_template(config.FINAL_ANSWER_PROMPT_FILE)

# --- Настройка Google AI ---
# --- Настройка OpenAI (OpenRouter) ---
# Используем httpx.Timeout для настройки таймаутов
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=config.OPENROUTER_API_KEY,
    http_client=httpx.Client(timeout=httpx.Timeout(60.0, connect=5.0)), # Общий таймаут 60с, таймаут подключения 5с
)
print("--- OpenAI SDK configured for OpenRouter via llm_interface. ---")

# --- Функции взаимодействия с LLM ---
def filter_query_with_llm_v2(user_question: str) -> dict:
    """
    Использует LLM (Gemini) для преобразования запроса пользователя в структурированные поисковые термины.
    Включает резервную логику и обработку таймаута.
    """
    print(f"  Filtering query with LLM ({config.LLM_MODEL_NAME}): '{user_question[:60]}...'")
    if not config.OPENROUTER_API_KEY or "YOUR_OPENROUTER_API_KEY" in config.OPENROUTER_API_KEY:
         return { "confirmation": f"Ошибка конфигурации AI. Ищу: '{user_question[:50]}...'", "semantic_query": user_question, "bm25_keywords": tokenize(user_question)[:5] }

    if "ERROR:" in SYSTEM_FILTER_PROMPT:  # Проверка: шаблон системы не загружен
         print(f"  ERROR: Cannot filter query, system prompt failed to load.")
         return { "confirmation": f"Ошибка загрузки системной инструкции AI. Ищу: '{user_question[:50]}...'", "semantic_query": user_question, "bm25_keywords": tokenize(user_question)[:5] }

    full_prompt = clean_text(SYSTEM_FILTER_PROMPT + "\nЗапрос пользователя: " + user_question)
    response = None
    content = ""
    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=config.generation_config["temperature"],
            max_tokens=2000, # Ограничение токенов для фильтрации
        )
        content = response.choices[0].message.content.strip()
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
    except httpx.TimeoutException as e:
        print(f"  ERROR: Timeout during filter_query_with_llm_v2: {e}")
        # Резервная логика
        return {
            "confirmation": f"Ошибка сети (таймаут) при обработке ИИ. Ищу по тексту: '{user_question[:50]}...'",
            "semantic_query": user_question,
            "bm25_keywords": tokenize(user_question)[:5]
        }
    except openai.APIError as e:
        print(f"  ERROR during filter_query_with_llm_v2 (OpenAI API Error): {e}")
        return {
            "confirmation": f"Произошла ошибка API при обработке запроса ИИ ({type(e).__name__}). Ищу по тексту: '{user_question[:50]}...'",
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
        if not config.OPENROUTER_API_KEY or "YOUR_OPENROUTER_API_KEY" in config.OPENROUTER_API_KEY:
             return "Ошибка: Ключ API для генерации ответа не настроен."

        cleaned_prompt = clean_text(prompt)

        response = client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[
                {"role": "user", "content": cleaned_prompt}
            ],
            temperature=config.generation_config["temperature"],
            max_tokens=1500, # Увеличиваем токены для генерации ответа
        )

        final_text = response.choices[0].message.content.strip()
        if not final_text:
            # OpenAI API обычно не возвращает пустой content, если нет ошибок.
            # Если это произошло, это может быть внутренняя проблема или очень короткий ответ.
            print(f"    WARNING: LLM final answer response content is empty.")
            final_text = f"Не удалось сгенерировать ответ (пустой ответ от модели)."

        print("  Final answer received from LLM.")
        return final_text

    # --- ОБРАБОТКА ОШИБКИ: таймаут ---
    except httpx.TimeoutException as e:
        print(f"  ERROR: Timeout during final answer generation call to LLM: {e}")
        return "Произошла ошибка: превышено время ожидания ответа от языковой модели при генерации ответа."
    except openai.APIError as e:
        print(f"  ERROR during final answer generation call to LLM (OpenAI API Error): {e}")
        return f"Произошла ошибка API ({type(e).__name__}) при генерации финального ответа. Пожалуйста, попробуйте позже."
    except Exception as e:
        print(f"  ERROR during final answer generation call to LLM: {type(e).__name__}: {e}")
        return f"Произошла внутренняя ошибка ({type(e).__name__}) при генерации финального ответа. Пожалуйста, попробуйте позже."
