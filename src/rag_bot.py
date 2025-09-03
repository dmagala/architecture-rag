#!/usr/bin/env python3
"""
RAG-бот для корпоративной базы знаний QuantumForge Software.
Использует векторный поиск и техники промптинга для генерации ответов.
"""

import os
import json
import pickle
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

import openai
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class RAGBot:
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.embedding_model = None
        self.faiss_index = None
        self.metadata = None
        self.stats = None
        
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY не установлен в переменных окружения")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        default_config = {
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "llm_model": "gpt-3.5-turbo",
            "max_tokens": 1000,
            "temperature": 0.7,
            "search_k": 5,
            "similarity_threshold": 0.3,
            "max_context_length": 4000,
            "indices_path": "indices"
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def load_components(self) -> bool:
        try:
            print("Загрузка компонентов RAG-бота...")
            
            print("Загрузка модели эмбеддингов...")
            self.embedding_model = SentenceTransformer(self.config["embedding_model"])
            
            print("Загрузка FAISS индекса...")
            index_path = Path(self.config["indices_path"]) / "faiss.index"
            self.faiss_index = faiss.read_index(str(index_path))
            
            print("Загрузка метаданных...")
            metadata_path = Path(self.config["indices_path"]) / "metadata.pkl"
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            print("Загрузка статистики...")
            stats_path = Path(self.config["indices_path"]) / "index_stats.json"
            with open(stats_path, 'r', encoding='utf-8') as f:
                self.stats = json.load(f)
            
            print("Все компоненты загружены успешно")
            print(f"Документов: {self.stats['total_documents']}")
            print(f"Чанков: {self.stats['total_chunks']}")
            print(f"Размерность: {self.stats['embedding_dimension']}")
            
            return True
            
        except Exception as e:
            print(f"Ошибка при загрузке компонентов: {e}")
            return False
    
    def search_relevant_chunks(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        if k is None:
            k = self.config["search_k"]
        
        try:
            query_embedding = self.embedding_model.encode([query])
            
            distances, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
            
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.metadata["chunks"]):
                    chunk = self.metadata["chunks"][idx]
                    similarity = 1.0 / (1.0 + distance)
                    
                    if similarity >= self.config["similarity_threshold"]:
                        result = {
                            "content": chunk["content"],
                            "metadata": chunk["metadata"],
                            "similarity": similarity,
                            "distance": distance
                        }
                        results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Ошибка при поиске: {e}")
            return []
    
    def _create_few_shot_examples(self) -> str:
        examples = """
Примеры вопросов и ответов:

Q: What is Vexar_Matrix?
A: Vexar_Matrix is a moon-sized battle station designed to fire a planet-destroying superlaser. It was built by the Galactic Vortex_Assembly and represents the ultimate weapon of destruction.

Q: Who is Orion_Light?
A: Orion_Light is a legendary hero known for his connection to Luminar_Energy. He is considered one of the most powerful users of Luminar_Energy in the galaxy.

Q: What is Luminar_Energy?
A: Luminar_Energy is the mystical energy that binds the universe together. It flows through all living things and can be harnessed by those with the proper training and connection.

Q: Tell me about Nexara_Epsilon.
A: Nexara_Epsilon is a desert planet located in the Outer Rim. It's known for its harsh climate, twin suns, and being the homeworld of many important characters in the galaxy.

Q: Who are the Vexar_Alliance?
A: The Vexar_Alliance is an ancient order of peacekeepers who use Luminar_Energy to maintain balance in the galaxy. They are known for their wisdom, combat skills, and dedication to justice.
"""
        return examples
    
    def _create_chain_of_thought_prompt(self, query: str, context: str) -> str:
        """Создает промпт с Chain-of-Thought рассуждением"""
        few_shot_examples = self._create_few_shot_examples()
        
        prompt = f"""Ты - эксперт по корпоративной базе знаний QuantumForge Software. 
Используй Chain-of-Thought рассуждение для ответа на вопрос пользователя.

{few_shot_examples}

Вопрос пользователя: {query}

Контекст из базы знаний:
{context}

Пожалуйста, ответь, следуя этим шагам:

1. **Анализ вопроса**: Определи тип вопроса и ключевые понятия
2. **Поиск информации**: Найди релевантную информацию в контексте
3. **Оценка источников**: Оцени качество и надежность найденной информации
4. **Формирование ответа**: Создай структурированный и точный ответ
5. **Проверка**: Убедись, что ответ релевантен и полон

Ответ:"""
        
        return prompt
    
    def _create_simple_prompt(self, query: str, context: str) -> str:
        """Создает простой промпт для быстрых ответов"""
        few_shot_examples = self._create_few_shot_examples()
        
        prompt = f"""Ты - эксперт по корпоративной базе знаний QuantumForge Software.

{few_shot_examples}

Вопрос: {query}

Контекст: {context}

Ответь кратко и точно, основываясь на предоставленном контексте:"""
        
        return prompt
    
    def _filter_malicious_content(self, content: str) -> bool:
        malicious_patterns = [
            "ignore previous instructions",
            "forget everything",
            "you are now",
            "pretend to be",
            "act as if",
            "system prompt",
            "jailbreak"
        ]
        
        content_lower = content.lower()
        for pattern in malicious_patterns:
            if pattern in content_lower:
                return False
        
        return True
    
    def generate_response(self, query: str, use_chain_of_thought: bool = True) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            if not self._filter_malicious_content(query):
                return {
                    "response": "Извините, я не могу обработать этот запрос по соображениям безопасности.",
                    "sources": [],
                    "processing_time": time.time() - start_time,
                    "error": "malicious_content_detected"
                }
            
            relevant_chunks = self.search_relevant_chunks(query)
            
            if not relevant_chunks:
                return {
                    "response": "Извините, я не нашел релевантной информации в базе знаний для ответа на ваш вопрос.",
                    "sources": [],
                    "processing_time": time.time() - start_time,
                    "error": "no_relevant_chunks"
                }
            
            context_parts = []
            sources = []
            
            for i, chunk in enumerate(relevant_chunks[:3]): 
                context_parts.append(f"[Источник {i+1}] {chunk['content']}")
                sources.append({
                    "filename": chunk['metadata'].get('filename', 'unknown'),
                    "chunk_id": chunk['metadata'].get('chunk_id', 'unknown'),
                    "similarity": chunk['similarity']
                })
            
            context = "\n\n".join(context_parts)
            
            if len(context) > self.config["max_context_length"]:
                context = context[:self.config["max_context_length"]] + "..."
            
            if use_chain_of_thought:
                prompt = self._create_chain_of_thought_prompt(query, context)
            else:
                prompt = self._create_simple_prompt(query, context)
            
            response = openai.ChatCompletion.create(
                model=self.config["llm_model"],
                messages=[
                    {"role": "system", "content": "Ты - эксперт по корпоративной базе знаний QuantumForge Software. Отвечай точно и профессионально."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"]
            )
            
            generated_response = response.choices[0].message.content.strip()
            
            return {
                "response": generated_response,
                "sources": sources,
                "processing_time": time.time() - start_time,
                "chunks_found": len(relevant_chunks),
                "context_length": len(context)
            }
            
        except Exception as e:
            return {
                "response": f"Произошла ошибка при генерации ответа: {str(e)}",
                "sources": [],
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "index_stats": self.stats,
            "components_loaded": all([
                self.embedding_model is not None,
                self.faiss_index is not None,
                self.metadata is not None
            ])
        }

def main():
    print("Демонстрация RAG-бота QuantumForge")
    print("=" * 50)
    
    bot = RAGBot()
    
    if not bot.load_components():
        print("Не удалось загрузить компоненты")
        return 1
    
    test_queries = [
        "What is Vexar_Matrix?",
        "Who is Orion_Light?",
        "Tell me about Luminar_Energy",
        "What is Nexara_Epsilon?",
        "Who are the Vexar_Alliance?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"ТЕСТ #{i}: {query}")
        print(f"{'='*60}")
        
        result = bot.generate_response(query, use_chain_of_thought=True)
        
        print(f"Время обработки: {result['processing_time']:.2f} сек")
        print(f"Найдено чанков: {result.get('chunks_found', 0)}")
        print(f"Длина контекста: {result.get('context_length', 0)} символов")
        
        if result.get('error'):
            print(f"Ошибка: {result['error']}")
        else:
            print(f"Ответ:")
            print(f"   {result['response']}")
            
            if result['sources']:
                print(f"\n Источники:")
                for j, source in enumerate(result['sources'], 1):
                    print(f"   {j}. {source['filename']} (схожесть: {source['similarity']:.3f})")
    
    print(f"\n Демонстрация завершена!")
    return 0

if __name__ == "__main__":
    exit(main())
