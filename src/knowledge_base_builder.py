#!/usr/bin/env python3
"""
Скрипт для создания уникальной базы знаний на основе Star Wars Wiki
с заменой всех ключевых терминов на вымышленные названия.
"""

import os
import json
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import random
from typing import Dict, List, Set
import uuid

class StarWarsKnowledgeBaseBuilder:
    def __init__(self, output_dir: str = "knowledge_base"):
        self.output_dir = output_dir
        self.terms_map = {}
        self.processed_urls = set()
        self.base_url = "https://starwars.fandom.com"
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.target_pages = [
            "/wiki/Luke_Skywalker",
            "/wiki/Darth_Vader",
            "/wiki/Princess_Leia",
            "/wiki/Han_Solo",
            "/wiki/Yoda",
            "/wiki/Obi-Wan_Kenobi",
            "/wiki/Emperor_Palpatine",
            "/wiki/Chewbacca",
            "/wiki/R2-D2",
            "/wiki/C-3PO",
            "/wiki/Death_Star",
            "/wiki/Lightsaber",
            "/wiki/Force",
            "/wiki/Jedi",
            "/wiki/Sith",
            "/wiki/Clone_Wars",
            "/wiki/Galactic_Empire",
            "/wiki/Rebel_Alliance",
            "/wiki/Tatooine",
            "/wiki/Coruscant",
            "/wiki/Dagobah",
            "/wiki/Hoth",
            "/wiki/Endor",
            "/wiki/Bespin",
            "/wiki/Naboo",
            "/wiki/Alderaan",
            "/wiki/Kashyyyk",
            "/wiki/Mustafar",
            "/wiki/Geonosis",
            "/wiki/Kamino",
            "/wiki/Republic",
            "/wiki/Separatists",
            "/wiki/Stormtrooper",
            "/wiki/X-wing",
            "/wiki/TIE_Fighter",
            "/wiki/Millennium_Falcon"
        ]
    
    def generate_fake_name(self, category: str) -> str:
        """Генерирует вымышленное имя для категории"""
        prefixes = {
            "character": ["Zara", "Kael", "Nyx", "Vex", "Rex", "Luna", "Orion", "Vera", "Kai", "Nova"],
            "planet": ["Zephyria", "Nexara", "Vortis", "Korvus", "Lumina", "Orionis", "Vexara", "Kaelos", "Nyxara", "Zephyron"],
            "technology": ["Quantum", "Nexus", "Vortex", "Korvex", "Luminar", "Orion", "Vexar", "Kaelon", "Nyxar", "Zephyr"],
            "organization": ["Quantum", "Nexus", "Vortex", "Korvex", "Luminar", "Orion", "Vexar", "Kaelon", "Nyxar", "Zephyr"],
            "weapon": ["Quantum", "Nexus", "Vortex", "Korvex", "Luminar", "Orion", "Vexar", "Kaelon", "Nyxar", "Zephyr"],
            "ship": ["Quantum", "Nexus", "Vortex", "Korvex", "Luminar", "Orion", "Vexar", "Kaelon", "Nyxar", "Zephyr"],
            "concept": ["Quantum", "Nexus", "Vortex", "Korvex", "Luminar", "Orion", "Vexar", "Kaelon", "Nyxar", "Zephyr"]
        }
        
        suffixes = {
            "character": ["Storm", "Blade", "Star", "Fire", "Wind", "Stone", "Light", "Dark", "Sky", "Earth"],
            "planet": ["Prime", "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota"],
            "technology": ["Core", "Matrix", "Grid", "Net", "Web", "Link", "Node", "Hub", "Gate", "Port"],
            "organization": ["Alliance", "Empire", "Republic", "Union", "Federation", "League", "Guild", "Order", "Council", "Assembly"],
            "weapon": ["Blade", "Strike", "Edge", "Point", "Tip", "Cut", "Slash", "Pierce", "Thrust", "Stab"],
            "ship": ["Cruiser", "Fighter", "Destroyer", "Battleship", "Corvette", "Frigate", "Carrier", "Transport", "Scout", "Patrol"],
            "concept": ["Force", "Power", "Energy", "Essence", "Spirit", "Soul", "Mind", "Heart", "Core", "Center"]
        }
        
        prefix = random.choice(prefixes.get(category, prefixes["concept"]))
        suffix = random.choice(suffixes.get(category, suffixes["concept"]))
        
        return f"{prefix}_{suffix}"
    
    def categorize_term(self, term: str) -> str:
        term_lower = term.lower()
        
        if any(name in term_lower for name in ["luke", "vader", "leia", "han", "yoda", "obi", "palpatine", "chewbacca", "r2", "c-3po"]):
            return "character"
        
        if any(name in term_lower for name in ["tatooine", "coruscant", "dagobah", "hoth", "endor", "bespin", "naboo", "alderaan", "kashyyyk", "mustafar", "geonosis", "kamino"]):
            return "planet"
        
        if any(name in term_lower for name in ["death star", "lightsaber", "x-wing", "tie fighter", "millennium falcon"]):
            return "technology"
        
        if any(name in term_lower for name in ["empire", "rebel", "jedi", "sith", "republic", "separatist"]):
            return "organization"
        
        if any(name in term_lower for name in ["lightsaber", "blaster", "saber"]):
            return "weapon"
        
        if any(name in term_lower for name in ["x-wing", "tie", "millennium", "falcon", "cruiser", "destroyer"]):
            return "ship"
        
        if any(name in term_lower for name in ["force", "clone wars", "stormtrooper"]):
            return "concept"
        
        return "concept"
    
    def create_terms_mapping(self) -> Dict[str, str]:
        terms_to_replace = [
            "Luke Skywalker", "Darth Vader", "Princess Leia", "Han Solo", "Yoda", 
            "Obi-Wan Kenobi", "Emperor Palpatine", "Chewbacca", "R2-D2", "C-3PO",
            "Anakin Skywalker", "Padmé Amidala", "Mace Windu", "Qui-Gon Jinn",
            "Count Dooku", "Darth Maul", "Jabba the Hutt", "Boba Fett",
            
            "Tatooine", "Coruscant", "Dagobah", "Hoth", "Endor", "Bespin", 
            "Naboo", "Alderaan", "Kashyyyk", "Mustafar", "Geonosis", "Kamino",
            "Jakku", "Takodana", "Starkiller Base", "Ahch-To",
            
            "Death Star", "Lightsaber", "X-wing", "TIE Fighter", "Millennium Falcon",
            "AT-AT", "AT-ST", "Star Destroyer", "Blaster", "Hyperdrive",
            "Holocron", "Kyber Crystal", "Droid", "Protocol Droid",
            
            "Jedi", "Sith", "Galactic Empire", "Rebel Alliance", "Republic",
            "Separatists", "Clone Wars", "Force", "Stormtrooper", "Clone Trooper",
            "Jedi Order", "Sith Order", "Dark Side", "Light Side",
            
            "Human", "Wookiee", "Droid", "Twi'lek", "Zabrak", "Togruta",
            "Mon Calamari", "Bothan", "Ewok", "Gungan", "Jawa", "Tusken Raider",
            
            "Clone Wars", "Galactic Civil War", "Battle of Yavin", "Battle of Endor",
            "Order 66", "Jedi Purge", "Sith Rule of Two", "Jedi Code"
        ]
        
        for term in terms_to_replace:
            if term not in self.terms_map:
                category = self.categorize_term(term)
                fake_name = self.generate_fake_name(category)
                self.terms_map[term] = fake_name
        
        return self.terms_map
    
    def download_page(self, url: str) -> str:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Ошибка при скачивании {url}: {e}")
            return ""
    
    def extract_text_content(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if not content_div:
            content_div = soup.find('div', {'class': 'mw-parser-output'})
        
        if content_div:
            for element in content_div.find_all(['div'], class_=re.compile(r'navbox|infobox|thumb')):
                element.decompose()
            
            text = content_div.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)
        
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def replace_terms(self, text: str) -> str:
        replaced_text = text
        
        sorted_terms = sorted(self.terms_map.items(), key=lambda x: len(x[0]), reverse=True)
        
        for original, replacement in sorted_terms:
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            replaced_text = pattern.sub(replacement, replaced_text)
        
        return replaced_text
    
    def save_document(self, title: str, content: str, filename: str):
        filepath = os.path.join(self.output_dir, f"{filename}.txt")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            f.write(content)
        
        print(f"Сохранен документ: {filepath}")
    
    def build_knowledge_base(self):
        print("Создание словаря замен терминов...")
        self.create_terms_mapping()
        
        print(f"Скачивание и обработка {len(self.target_pages)} страниц...")
        
        for i, page_path in enumerate(self.target_pages, 1):
            url = urljoin(self.base_url, page_path)
            print(f"[{i}/{len(self.target_pages)}] Обработка: {url}")
            
            html = self.download_page(url)
            if not html:
                continue
            
            text_content = self.extract_text_content(html)
            if not text_content:
                continue
            
            replaced_content = self.replace_terms(text_content)
            
            page_name = page_path.split('/')[-1].replace('_', ' ').title()
            filename = f"document_{i:02d}_{page_name.replace(' ', '_').lower()}"
            
            self.save_document(page_name, replaced_content, filename)
            
            time.sleep(1)
        
        terms_map_path = os.path.join(self.output_dir, "terms_map.json")
        with open(terms_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.terms_map, f, ensure_ascii=False, indent=2)
        
        print(f"\nБаза знаний создана в директории: {self.output_dir}")
        print(f"Создано документов: {len(self.target_pages)}")
        print(f"Словарь замен сохранен в: {terms_map_path}")
        print(f"Всего заменено терминов: {len(self.terms_map)}")

def main():
    builder = StarWarsKnowledgeBaseBuilder()
    builder.build_knowledge_base()

if __name__ == "__main__":
    main()
