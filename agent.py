import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import datetime
import json
import numpy as np
import uuid
from cachetools import TTLCache
import sqlite3
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional, Any, Callable
import together
import time
from sympy import sympify, integrate, sin, pi
import re
import os


# Получение API-ключей из переменных окружения
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if DEEPSEEK_API_KEY:
    os.environ["TOGETHER_API_KEY"] = DEEPSEEK_API_KEY
WOLFRAM_APP_ID = os.environ.get("WOLFRAM_APP_ID", "")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")

# =====================================================================
# Инструменты
# =====================================================================

calculate_cache = TTLCache(maxsize=1000, ttl=86400)  # Кэш на 24 часа
search_cache = TTLCache(maxsize=1000, ttl=86400)  # Кэш на 24 часа

def calculate(expression: str, kwargs: dict = None) -> str:
    """
    Выполняет аналитические вычисления через Wolfram Alpha или локально с sympy.
    Возвращает ответ в формате: <think>рассуждения</think>итоговый_ответ
    """
    if expression in calculate_cache:
        return calculate_cache[expression]

    if WOLFRAM_APP_ID:
        url = "http://api.wolframalpha.com/v2/query"
        params = {"appid": WOLFRAM_APP_ID, "input": expression, "output": "XML"}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            for pod in root.findall(".//pod[@id='Result']"):
                plaintext = pod.find("subpod/plaintext")
                if plaintext is not None and plaintext.text:
                    result = plaintext.text.strip()
                    think = f"Выполнено вычисление через Wolfram Alpha: {expression} = {result}"
                    final_result = f"<think>{think}</think>{result}"
                    calculate_cache[expression] = final_result
                    return final_result
            return "<think>Wolfram Alpha не смог дать однозначный ответ.</think>Не удалось вычислить."
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                return "<think>Превышен лимит запросов к Wolfram Alpha.</think>Ошибка: лимит запросов."
            return f"<think>Ошибка Wolfram Alpha: {str(e)}.</think>Пытаемся локально."

    try:
        expr = sympify(expression, evaluate=False)
        result = expr.doit() if expr.is_Integral else expr.evalf()
        think = f"Локальное вычисление с sympy: {expression} = {result}"
        final_result = f"<think>{think}</think>{result}"
        calculate_cache[expression] = final_result
        return final_result
    except Exception as e:
        return f"<think>Ошибка локального вычисления: {str(e)}</think>Не удалось вычислить."

def numerical_calc(expression: str, kwargs: dict = None) -> str:
    """
    Выполняет численные вычисления с использованием sympy.
    Возвращает ответ в формате: <think>рассуждения</think>итоговый_ответ
    """
    try:
        expr = sympify(expression, evaluate=False)
        result = expr.doit().evalf() if expr.is_Integral else expr.evalf()
        think = f"Численное вычисление с sympy: {expression} = {result}"
        return f"<think>{think}</think>{result}"
    except Exception as e:
        return f"<think>Ошибка численного вычисления: {str(e)}</think>Не удалось вычислить."

encoder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def web_search(query: str, kwargs: dict = None) -> str:
    """
    Расширенный интернет-поиск с использованием SerpApi и семантического ранжирования.
    Возвращает ответ в формате: <think>рассуждения</think>итоговый_ответ
    """
    serpapi_key = kwargs.get("serpapi_key", os.environ.get("SERPAPI_KEY"))
    if query in search_cache:
        return search_cache[query]

    if not serpapi_key:
        return "<think>Требуется ключ SerpApi.</think>Ошибка: отсутствует ключ API."

    try:
        url = "https://serpapi.com/search"
        params = {"q": query, "api_key": serpapi_key, "num": 5}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get("organic_results", [])
        if not results:
            return "<think>Поиск через SerpApi не вернул результатов.</think>По вашему запросу ничего не найдено."

        query_embedding = encoder.encode(query)
        scored_results = []
        think_steps = ["Выполнен поиск через SerpApi:"]
        for result in results:
            url = result.get("link")
            snippet = result.get("snippet", "")
            if not snippet:
                try:
                    page_response = requests.get(url, timeout=10)
                    page_response.raise_for_status()
                    soup = BeautifulSoup(page_response.text, "html.parser")
                    paragraphs = soup.find_all("p")
                    snippet = " ".join(p.text.strip() for p in paragraphs)[:1000]
                    think_steps.append(f"Извлечён текст из {url}")
                except Exception as e:
                    think_steps.append(f"Ошибка загрузки {url}: {str(e)}")
                    continue
            page_embedding = encoder.encode(snippet)
            similarity = np.dot(query_embedding, page_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(page_embedding)
            )
            scored_results.append((url, snippet, similarity, result.get("title", "")))
            think_steps.append(f"Схожесть для {url}: {similarity:.2f}")

        if not scored_results:
            return "<think>Не удалось получить релевантные результаты.</think>Не удалось получить результаты."

        scored_results.sort(key=lambda x: x[2], reverse=True)
        output_lines = []
        for url, snippet, score, title in scored_results[:3]:
            output_lines.append(f"{title}\n{url}\nСхожесть: {score:.2f}\nОтрывок: {snippet[:200]}...")
        result = f"<think>{' '.join(think_steps)}</think>Лучшие результаты поиска:\n" + "\n\n".join(output_lines)
        search_cache[query] = result
        return result
    except Exception as e:
        return f"<think>Ошибка поиска: {str(e)}</think>Не удалось выполнить поиск."

# =====================================================================
# Классы для логирования, памяти и инструментов
# =====================================================================

class ExecutionLogger:
    """Структурированное логирование событий агента с сохранением в SQLite."""
    def __init__(self, session_id: Optional[str] = None, db_path: str = "agent_logs.db"):
        self.session_id = session_id or str(uuid.uuid4())
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    session_id TEXT,
                    agent TEXT,
                    event TEXT,
                    thoughts TEXT,
                    details TEXT
                )
            """)

    def log(self, agent: str, event: str, details: Dict, thoughts: str = ""):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": self.session_id,
            "agent": agent,
            "event": event,
            "thoughts": thoughts,
            "details": json.dumps(details, ensure_ascii=False)
        }
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO logs (timestamp, session_id, agent, event, thoughts, details) VALUES (?, ?, ?, ?, ?, ?)",
                (entry["timestamp"], entry["session_id"], entry["agent"], entry["event"], entry["thoughts"], entry["details"])
            )
            conn.commit()
        print(json.dumps(entry, indent=2, ensure_ascii=False))

    def export_log(self, event: Optional[str] = None) -> str:
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM logs"
            if event:
                query += " WHERE event = ?"
                logs = conn.execute(query, (event,)).fetchall()
            else:
                logs = conn.execute(query).fetchall()
        return json.dumps(
            [{"timestamp": row[1], "session_id": row[2], "agent": row[3], "event": row[4], "thoughts": row[5], "details": json.loads(row[6])} for row in logs],
            indent=2, ensure_ascii=False
        )

class VectorMemory:
    """Векторная память с использованием SentenceTransformer и SQLite."""
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2', db_path: str = "memory.db"):
        self.model = SentenceTransformer(model_name)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT,
                    vector BLOB,
                    timestamp TEXT
                )
            """)

    def add(self, text: str):
        embedding = self.model.encode(text)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO memory (text, vector, timestamp) VALUES (?, ?, ?)",
                (text, embedding.tobytes(), datetime.datetime.now().isoformat())
            )
            conn.commit()

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        query_vec = self.model.encode(query)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT text, vector FROM memory").fetchall()
        scored = []
        for text, vec_bytes in rows:
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            score = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
            scored.append((text, score))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

class Tool:
    """Базовый класс инструмента для использования агентом."""
    def __init__(self, name: str, func: Callable[[str, dict], str], description: str, input_type: type = str):
        self.name = name
        self.func = func
        self.description = description
        self.input_type = input_type

    def execute(self, input_data: Any, **kwargs) -> str:
        if not isinstance(input_data, self.input_type):
            return f"<think>Ошибка: Ожидается входной тип {self.input_type}, получен {type(input_data)}</think>Ошибка типа данных."
        try:
            return self.func(input_data, kwargs)
        except Exception as e:
            return f"<think>Ошибка выполнения инструмента {self.name}: {str(e)}</think>Не удалось выполнить."

# =====================================================================
# DeepSeekAgent
# =====================================================================

MAX_SIZE = 4096  # Уменьшено для соответствия лимитам Together API
Temperature = 0.6  # Рекомендуемое значение для DeepSeek-R1

class DeepSeekAgent:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"):
        self.client = together.Together(api_key=os.environ["TOGETHER_API_KEY"])
        self.model = model_name
        self.cache = TTLCache(maxsize=10000, ttl=3600)  # Кэш на 1 час

    def generate_response(self, prompt: str, max_tokens: int = MAX_SIZE, temperature: float = Temperature, system_prompt: str = None) -> str:
        cache_key = (prompt, max_tokens, temperature, system_prompt)
        if cache_key in self.cache:
            return self.cache[cache_key]

        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        for attempt in range(3):  # Повторные попытки
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                result = response.choices[0].message.content
                self.cache[cache_key] = result
                return result
            except Exception as e:
                if attempt == 2:
                    return f"<think>Ошибка LLM после 3 попыток: {str(e)}</think>Ошибка LLM."
                time.sleep(1)  # Пауза перед повторной попыткой

# =====================================================================
# BaseAgent
# =====================================================================

def parse_response(response: str) -> tuple[str, str]:
    """
    Извлекает рассуждения и итоговый ответ из текста.
    Возвращает кортеж: (рассуждения, итоговый ответ).
    """
    think_pattern = r'<think>(.*?)</think>(.*)'
    match = re.match(think_pattern, response, re.DOTALL)
    if match:
        thoughts, final_answer = match.groups()
        return thoughts.strip(), final_answer.strip()
    return "", response.strip()  # Если тегов нет, считаем всё ответом

def parse_tool_choices(response: str) -> List[str]:
    """
    Извлекает список строк в формате ИНСТРУМЕНТ|ВВОД из ответа, игнорируя <think>.
    """
    thoughts, answer = parse_response(response)
    return [line.strip() for line in answer.split('\n') if line.strip() and '|' in line]

class BaseAgent:
    """Базовый агент с поддержкой составных задач и отделения рассуждений."""
    def __init__(self, name: str, llm, tools: Optional[List[Tool]] = None, logger: Optional[ExecutionLogger] = None):
        self.name = name
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools or []}
        self.logger = logger or ExecutionLogger()
        self.memory = VectorMemory()
        self.history = []  # Хранит только итоговые ответы

    def log(self, event: str, details: dict, thoughts: str = ""):
        self.logger.log(self.name, event, details, thoughts)

    def solve_task(self, task: str, memory_threshold: float = 0.9) -> str:
        self.log("TASK_RECEIVED", {"task": task})

        # Проверяем память
        similar = self.memory.search(task)
        if similar and similar[0][1] > memory_threshold:
            self.log("MEMORY_HIT", {"record": similar[0][0], "score": similar[0][1]})
            return f"(Из памяти): {similar[0][0]}"

        # Формируем историю (только итоговые ответы)
        context = "\n".join([f"Запрос: {h['task']}\nОтвет: {h['result']}" for h in self.history[-3:]])

        # Промпт для выбора инструмента
        tool_list = ", ".join([f"{name} ({tool.description})" for name, tool in self.tools.items()])
        tool_prompt = (
            f"Ты — эксперт в выборе инструментов. Доступные инструменты: {tool_list}. "
            f"Для задачи '{task}' определи, какие инструменты нужны для её выполнения. "
            f"Верни список строк в формате ИНСТРУМЕНТ|ВВОД, по одной строке на инструмент. "
            f"Если требуется только один инструмент, верни одну строку. "
            f"Если ни один инструмент не подходит, верни 'LLM'. "
            f"Для интегралов используй формат sympy, например, 'integrate(sin(x), (x, 0, pi))' для аналитического и численного вычислений. "
            f"Для веб-поиска используй точный запрос, например, 'integral sin(x) from 0 to pi'. "
            f"Примеры:\n"
            f"- Для 'Найти информацию о Python' вернуть 'web|Python'.\n"
            f"- Для 'Рассчитай 2+2' вернуть 'calculate|2+2'.\n"
            f"- Для 'Объясни квантовую механику' вернуть 'LLM'.\n"
            f"- Для 'Рассчитай интеграл от sin(x) от 0 до pi аналитически и численно, а также найди итоговый ответ в интернете' вернуть:\n"
            f"  calculate|integrate(sin(x), (x, 0, pi))\n  numerical|integrate(sin(x), (x, 0, pi))\n  web|integral sin(x) from 0 to pi\n"
            f"Контекст:\n{context}\n"
            f"Формат ответа: <think>Рассуждения</think>строка1\nстрока2\n..."
        )
        system_prompt = "Ты — логичный и точный ассистент. Отвечай на русском языке."
        tool_response = self.llm.generate_response(tool_prompt, max_tokens=512, system_prompt=system_prompt)
        thoughts, tool_choices = parse_response(tool_response)
        
        # Сохраняем только итоговые строки в details.response
        self.log("TOOL_CHOICE", {"response": tool_choices, "parsed_choices": parse_tool_choices(tool_response)}, thoughts=thoughts)

        # Если выбран только LLM
        if tool_choices.strip() == "LLM":
            deep_think_prompt = (
                f"Ты — эксперт в математике, программировании и логике. Реши задачу '{task}' на русском языке. "
                f"Рассуждай шаг за шагом, проверяя свои выводы. Контекст:\n{context}"
            )
            system_prompt = (
                "Ты — эксперт в математике, программировании и логике. "
                "Для каждой задачи предоставляй ответ в формате: <think>Твои пошаговые рассуждения</think>Итоговый ответ. "
                "Отвечай на русском языке."
            )
            response = self.llm.generate_response(deep_think_prompt, max_tokens=1024, system_prompt=system_prompt)
            thoughts, final_answer = parse_response(response)
            self.memory.add(final_answer)
            self.history.append({"task": task, "result": final_answer})
            self.log("LLM_RESULT", {"result": final_answer}, thoughts=thoughts)
            return final_answer

        # Обработка списка инструментов
        results = []
        parsed_choices = parse_tool_choices(tool_response)
        if not parsed_choices:
            self.log("TOOL_PARSING_ERROR", {"response": tool_response})
            return "Ошибка: Неверный формат ответа выбора инструмента."

        for choice in parsed_choices:
            try:
                tool_name, input_data = [s.strip() for s in choice.split("|", 1)]
                if tool_name in self.tools:
                    result = self.tools[tool_name].execute(input_data, serpapi_key=os.environ.get("SERPAPI_KEY"))
                    thoughts, final_answer = parse_response(result)
                    self.memory.add(final_answer)
                    self.history.append({"task": f"{task} ({tool_name})", "result": final_answer})
                    self.log("TOOL_EXECUTED", {"tool": tool_name, "input": input_data, "result": final_answer}, thoughts=thoughts)
                    results.append(f"{tool_name}: {final_answer}")
                else:
                    self.log("TOOL_NOT_FOUND", {"tool": tool_name})
                    results.append(f"Инструмент {tool_name} не найден.")
            except ValueError:
                self.log("TOOL_PARSING_ERROR", {"choice": choice})
                results.append(f"Ошибка парсинга: {choice}")

        return "\n".join(results)

# =====================================================================
# Пример использования
# =====================================================================

# Создаём инструменты
tools = [
    Tool("calculate", calculate, "Аналитические вычисления с Wolfram Alpha или sympy", str),
    Tool("numerical", numerical_calc, "Численные вычисления с sympy", str),
    Tool("web", web_search, "Расширенный интернет-поиск с семантическим ранжированием", str)
]

# Инициализируем агента
llm = DeepSeekAgent()
logger = ExecutionLogger()
agent = BaseAgent("Интеллектуальный Агент", llm, tools, logger)

# Выполняем задачу
task = "Рассчитай интеграл от sin(x) от 0 до pi аналитически и численно, а также найди итоговый ответ в интернете"
result = agent.solve_task(task)
print("Итоговый результат работы агента:")
print(result)
print("\nЛоги:")
print(logger.export_log())