#!/usr/bin/env python3
"""
Скрипт для создания векторного индекса базы знаний.
Использует выбранную модель эмбеддингов (bge-base-en) и FAISS для индексации.
"""

import os
import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from sentence_transformers import SentenceTransformer

class VectorIndexBuilder:
    def __init__(self, 
                 knowledge_base_path: str = "knowledge_base",
                 index_path: str = "indices",
                 embedding_model: str = "BAAI/bge-base-en-v1.5"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.index_path = Path(index_path)
        self.embedding_model_name = embedding_model
        
        self.index_path.mkdir(exist_ok=True)
        
        self.embedding_model = None
        self.text_splitter = None
        self.vector_store = None
        
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "embedding_dimension": 0,
            "index_size": 0,
            "processing_time": 0
        }
    
    def load_embedding_model(self):
        print(f"Загрузка модели эмбеддингов: {self.embedding_model_name}")
        
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            test_embedding = self.embedding_model.encode(["test"])
            self.stats["embedding_dimension"] = test_embedding.shape[1]
            
            print(f"✓ Модель загружена. Размерность эмбеддингов: {self.stats['embedding_dimension']}")
            
        except Exception as e:
            print(f"✗ Ошибка загрузки модели: {e}")
            raise
    
    def setup_text_splitter(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        print(f"Настройка разделителя текста (chunk_size={chunk_size}, overlap={chunk_overlap})")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        print("✓ Разделитель текста настроен")
    
    def load_documents(self) -> List[Document]:
        """Загружает и обрабатывает документы из базы знаний"""
        print("Загрузка документов из базы знаний...")
        
        documents = []
        txt_files = list(self.knowledge_base_path.glob("*.txt"))
        
        excluded_files = {"malicious_document.txt", "benign_document.txt"}
        txt_files = [f for f in txt_files if f.name not in excluded_files]
        
        print(f"Найдено {len(txt_files)} документов для обработки")
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "file_type": "text"
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                print(f"✗ Ошибка загрузки {file_path}: {e}")
        
        self.stats["total_documents"] = len(documents)
        print(f"✓ Загружено {len(documents)} документов")
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Разбивает документы на чанки"""
        print("Разбиение документов на чанки...")
        
        if not self.text_splitter:
            raise ValueError("Разделитель текста не настроен")
        
        chunks = self.text_splitter.split_documents(documents)
        
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk.page_content)
            })
        
        self.stats["total_chunks"] = len(chunks)
        print(f"✓ Создано {len(chunks)} чанков")
        
        return chunks
    
    def create_embeddings(self, chunks: List[Document]) -> np.ndarray:
        print("Создание эмбеддингов...")
        
        if not self.embedding_model:
            raise ValueError("Модель эмбеддингов не загружена")
        
        texts = [chunk.page_content for chunk in chunks]
        
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts, show_progress_bar=True)
            all_embeddings.append(batch_embeddings)
            
            print(f"Обработано {min(i + batch_size, len(texts))}/{len(texts)} чанков")
        
        embeddings = np.vstack(all_embeddings)
        
        print(f"✓ Создано {embeddings.shape[0]} эмбеддингов размерности {embeddings.shape[1]}")
        
        return embeddings
    
    def create_faiss_index(self, chunks: List[Document], embeddings: np.ndarray):
        print("Создание FAISS индекса...")
        
        import faiss
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        index.add(embeddings.astype('float32'))
        
        index_file = self.index_path / "faiss.index"
        faiss.write_index(index, str(index_file))
        
        metadata = {
            "chunks": [
                {
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                }
                for chunk in chunks
            ],
            "stats": self.stats
        }
        
        metadata_file = self.index_path / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        self.vector_store = index
        self.stats["index_size"] = index.ntotal
        
        print(f"✓ FAISS индекс создан и сохранен в {index_file}")
        print(f"✓ Метаданные сохранены в {metadata_file}")
    
    def test_index(self, test_queries: List[str] = None):
        if test_queries is None:
            test_queries = [
                "Who is Orion_Light?",
                "What is Vexar_Matrix?",
                "Tell me about Nexara_Epsilon",
                "What is Luminar_Energy?",
                "Who are the Vexar_Alliance?"
            ]
        
        print("Тестирование индекса...")
        
        if not self.vector_store:
            raise ValueError("Индекс не создан")
        
        metadata_file = self.index_path / "metadata.pkl"
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        for query in test_queries:
            print(f"\nЗапрос: {query}")
            
            query_embedding = self.embedding_model.encode([query])
            
            k = 3
            distances, indices = self.vector_store.search(query_embedding.astype('float32'), k)
            
            print("Найденные чанки:")
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                chunk = metadata["chunks"][idx]
                content_preview = chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"]
                print(f"  {i+1}. [расстояние: {distance:.3f}] {content_preview}")
    
    def build_index(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        start_time = time.time()
        
        print("=" * 60)
        print("СОЗДАНИЕ ВЕКТОРНОГО ИНДЕКСА")
        print("=" * 60)
        
        try:
            self.load_embedding_model()
            
            self.setup_text_splitter(chunk_size, chunk_overlap)
            
            documents = self.load_documents()
            
            chunks = self.split_documents(documents)
            
            embeddings = self.create_embeddings(chunks)
            
            self.create_faiss_index(chunks, embeddings)
            
            self.test_index()
            
            self.stats["processing_time"] = time.time() - start_time
            
            stats_file = self.index_path / "index_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
            
            print("\n" + "=" * 60)
            print("ИНДЕКС СОЗДАН УСПЕШНО!")
            print("=" * 60)
            print(f"Документов обработано: {self.stats['total_documents']}")
            print(f"Чанков создано: {self.stats['total_chunks']}")
            print(f"Размерность эмбеддингов: {self.stats['embedding_dimension']}")
            print(f"Размер индекса: {self.stats['index_size']}")
            print(f"Время выполнения: {self.stats['processing_time']:.2f} секунд")
            print(f"Файлы сохранены в: {self.index_path}")
            
        except Exception as e:
            print(f"✗ Ошибка при создании индекса: {e}")
            raise

def main():
    builder = VectorIndexBuilder()
    builder.build_index()

if __name__ == "__main__":
    main()
