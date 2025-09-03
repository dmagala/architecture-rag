#!/usr/bin/env python3
"""
FastAPI интерфейс для RAG-бота QuantumForge Software.
Предоставляет REST API для взаимодействия с ботом.
"""

import os
import time
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from rag_bot import RAGBot

# Модели данных
class QueryRequest(BaseModel):
    """Модель запроса"""
    query: str = Field(..., description="Вопрос пользователя", max_length=500)
    use_chain_of_thought: bool = Field(True, description="Использовать Chain-of-Thought")
    max_sources: int = Field(5, description="Максимальное количество источников", ge=1, le=10)

class QueryResponse(BaseModel):
    """Модель ответа"""
    response: str
    sources: list
    processing_time: float
    chunks_found: int
    context_length: int
    timestamp: str
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Модель статуса здоровья"""
    status: str
    timestamp: str
    components_loaded: bool
    index_stats: Dict[str, Any]

# Создаем FastAPI приложение
app = FastAPI(
    title="QuantumForge RAG Bot API",
    description="API для интеллектуального бота корпоративной базы знаний",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальная переменная для бота
rag_bot: Optional[RAGBot] = None

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    global rag_bot
    
    print("🚀 Запуск QuantumForge RAG Bot API...")
    
    try:
        # Создаем и инициализируем бота
        rag_bot = RAGBot()
        
        # Загружаем компоненты
        if not rag_bot.load_components():
            raise Exception("Не удалось загрузить компоненты RAG-бота")
        
        print("✅ RAG-бот успешно инициализирован")
        
    except Exception as e:
        print(f"❌ Ошибка при инициализации: {e}")
        raise

@app.get("/", response_model=Dict[str, str])
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "QuantumForge RAG Bot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка состояния API"""
    if not rag_bot:
        raise HTTPException(status_code=503, detail="RAG-бот не инициализирован")
    
    stats = rag_bot.get_stats()
    
    return HealthResponse(
        status="healthy" if stats["components_loaded"] else "unhealthy",
        timestamp=datetime.now().isoformat(),
        components_loaded=stats["components_loaded"],
        index_stats=stats.get("index_stats", {})
    )

@app.post("/query", response_model=QueryResponse)
async def query_bot(request: QueryRequest):
    """Основной эндпоинт для запросов к боту"""
    if not rag_bot:
        raise HTTPException(status_code=503, detail="RAG-бот не инициализирован")
    
    try:
        # Генерируем ответ
        result = rag_bot.generate_response(
            query=request.query,
            use_chain_of_thought=request.use_chain_of_thought
        )
        
        # Ограничиваем количество источников
        if len(result["sources"]) > request.max_sources:
            result["sources"] = result["sources"][:request.max_sources]
        
        # Добавляем временную метку
        result["timestamp"] = datetime.now().isoformat()
        
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}")

@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """Получение статистики бота"""
    if not rag_bot:
        raise HTTPException(status_code=503, detail="RAG-бот не инициализирован")
    
    return rag_bot.get_stats()

@app.post("/search")
async def search_chunks(query: str, k: int = 5):
    """Поиск релевантных чанков без генерации ответа"""
    if not rag_bot:
        raise HTTPException(status_code=503, detail="RAG-бот не инициализирован")
    
    try:
        chunks = rag_bot.search_relevant_chunks(query, k=k)
        
        return {
            "query": query,
            "chunks_found": len(chunks),
            "chunks": [
                {
                    "content": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                    "filename": chunk["metadata"].get("filename", "unknown"),
                    "similarity": chunk["similarity"]
                }
                for chunk in chunks
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при поиске: {str(e)}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Обработчик HTTP исключений"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Обработчик общих исключений"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Внутренняя ошибка сервера",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

def main():
    """Запуск API сервера"""
    print("🚀 Запуск QuantumForge RAG Bot API...")
    
    # Проверяем наличие API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Ошибка: OPENAI_API_KEY не установлен")
        print("Установите переменную окружения: export OPENAI_API_KEY='your-key'")
        return 1
    
    # Запускаем сервер
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    exit(main())
