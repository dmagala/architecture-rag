#!/usr/bin/env python3
"""
Скрипт для запуска RAG-бота QuantumForge Software.
Поддерживает различные режимы работы: демо, API, тестирование.
"""

import os
import sys
import argparse
import time
from pathlib import Path

sys.path.append("src")

def check_requirements():
    print("Проверка требований...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Ошибка: OPENAI_API_KEY не установлен")
        print("Установите переменную окружения: export OPENAI_API_KEY='your-key'")
        return False
    
    index_files = [
        "indices/faiss.index",
        "indices/metadata.pkl", 
        "indices/index_stats.json"
    ]
    
    for file_path in index_files:
        if not Path(file_path).exists():
            print(f"Отсутствует файл: {file_path}")
            print("Сначала создайте векторный индекс: python build_vector_index.py")
            return False
    
    if not Path("config.json").exists():
        print("Отсутствует файл конфигурации: config.json")
        return False
    
    print("Все требования выполнены")
    return True

def run_demo():
    print("Запуск демонстрации RAG-бота")
    print("=" * 50)
    
    try:
        from rag_bot import RAGBot
        
        bot = RAGBot()
        
        if not bot.load_components():
            print("Не удалось загрузить компоненты")
            return 1
        
        print("RAG-бот готов к работе!")
        print("\n Примеры запросов:")
        print("   - What is Vexar_Matrix?")
        print("   - Who is Orion_Light?")
        print("   - Tell me about Luminar_Energy")
        print("   - What is Nexara_Epsilon?")
        print("   - Who are the Vexar_Alliance?")
        print("\nВведите 'quit' для выхода")
        
        while True:
            try:
                query = input("\n Ваш вопрос: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("До свидания!")
                    break
                
                if not query:
                    continue
                
                print("Обрабатываю запрос...")
                start_time = time.time()
                
                result = bot.generate_response(query, use_chain_of_thought=True)
                
                processing_time = time.time() - start_time
                
                print(f"\n️  Время обработки: {processing_time:.2f} сек")
                print(f"Найдено чанков: {result.get('chunks_found', 0)}")
                
                if result.get('error'):
                    print(f"Ошибка: {result['error']}")
                else:
                    print(f"\n Ответ:")
                    print(f"   {result['response']}")
                    
                    if result.get('sources'):
                        print(f"\n Источники:")
                        for i, source in enumerate(result['sources'][:3], 1):
                            print(f"   {i}. {source['filename']} (схожесть: {source['similarity']:.3f})")
                
            except KeyboardInterrupt:
                print("\n До свидания!")
                break
            except Exception as e:
                print(f"Ошибка: {e}")
        
        return 0
        
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("Установите зависимости: pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"Ошибка: {e}")
        return 1

def run_api():
    """Запускает API сервер"""
    print("Запуск API сервера")
    print("=" * 50)
    
    try:
        from api import main as api_main
        return api_main()
        
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("Установите зависимости: pip install fastapi uvicorn")
        return 1
    except Exception as e:
        print(f"Ошибка: {e}")
        return 1

def run_tests():
    """Запускает тестирование"""
    print("Запуск тестирования")
    print("=" * 50)
    
    try:
        from test_rag_bot import main as test_main
        return test_main()
        
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("Установите зависимости: pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"Ошибка: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description="RAG-бот QuantumForge Software",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python run_rag_bot.py demo          # Интерактивная демонстрация
  python run_rag_bot.py api           # Запуск API сервера
  python run_rag_bot.py test          # Запуск тестов
  python run_rag_bot.py --help        # Показать справку
        """
    )
    
    parser.add_argument(
        'mode',
        nargs='?',
        choices=['demo', 'api', 'test'],
        help='Режим работы: demo (демонстрация), api (сервер), test (тестирование)'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Только проверить требования, не запускать'
    )
    
    args = parser.parse_args()
    
    if not check_requirements():
        return 1
    
    if args.check_only:
        print("Проверка завершена успешно")
        return 0
    
    if not args.mode:
        parser.print_help()
        return 0
    
    if args.mode == 'demo':
        return run_demo()
    elif args.mode == 'api':
        return run_api()
    elif args.mode == 'test':
        return run_tests()
    
    return 0

if __name__ == "__main__":
    exit(main())
