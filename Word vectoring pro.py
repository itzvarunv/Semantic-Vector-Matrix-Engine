import json
import random
import os
import math

MEMORY_FILE = "semantic_brain.json"
HISTORY_FILE = "relations_history.json"
DIMENSIONS = 15
LEARNING_SPEED = 0.02  # A small shift so it takes multiple contexts to confidently converge
ledger = {}
suspected_stop_words = {}
MAX_UNIQUE_RELATIONS = 100 # it's clearly a cross-niche structural word (garbage glue text).

def load_brain_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
                word_vectors = memory_data.get("word_vectors", {})
                known_words = memory_data.get("known_words", {})
                return word_vectors, known_words
                
        except json.JSONDecodeError:
            print("[!] Warning: 'semantic_brain.json' was corrupted. Initializing fresh.")
            return {}, {}
    else:
        print("[!] No prior memory state detected. Initializing fresh database with seed rules.")
        initial_known_words = {
            "car": "Acceptable",
            "engine": "Acceptable",
            "crankshaft": "Acceptable",
            "the": "Unacceptable",
            "is": "Unacceptable",
            "and": "Unacceptable",
            "a": "Unacceptable"
        }
        
        return {}, initial_known_words

def save_brain_memory(word_vectors, known_words):
    memory_payload = {
        "word_vectors": word_vectors,
        "known_words": known_words
    }
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory_payload, f, indent=4)
    print(f"[+] Operational vectors and rules written safely to '{MEMORY_FILE}'.")

def vectorization(word, word_vectors):
    random_coordinates = [round(random.uniform(-1.0, 1.0), 4) for _ in range(DIMENSIONS)]
    word_vectors[word] = random_coordinates
    return word_vectors

def converge_vectors(word1, word2, word_vectors):
    vec1 = word_vectors[word1]
    vec2 = word_vectors[word2]
    for i in range(DIMENSIONS):
        gap = vec2[i] - vec1[i]
        vec1[i] = round(vec1[i] + (LEARNING_SPEED * gap), 4)
        vec2[i] = round(vec2[i] - (LEARNING_SPEED * gap), 4)
    return word_vectors

def ledger_update(word1, word2):
    global ledger, suspected_stop_words
    if word1 not in ledger:
        ledger[word1] = set() # Set ensures unique connections only
    ledger[word1].add(word2)
    if len(ledger[word1]) > MAX_UNIQUE_RELATIONS:
        suspected_stop_words[word1] = len(ledger[word1])
    if word2 not in ledger:
        ledger[word2] = set()
    ledger[word2].add(word1)
    if len(ledger[word2]) > MAX_UNIQUE_RELATIONS:
        suspected_stop_words[word2] = len(ledger[word2])

def review_stop_words(word_vectors, known_words):
    global suspected_stop_words
    if not suspected_stop_words:
        print("\n[i] Stop-Word Analysis: No words behaved like garbage glue text.")
        return word_vectors, known_words
        
    print("\n=======================================================")
    print(" STOP-WORD CANDIDATES DETECTED FOR YOUR APPROVAL")
    print("The machine thinks these words connect to too many random topics:")
    print("=======================================================")
    for word, connection_rate in list(suspected_stop_words.items()):
        print(f"\nWord: '{word}' -> Linked to {connection_rate}% of your total vocabulary.")
        choice = input(f"Ban '{word}' and treat as a Stop-Word? (y/n): ").strip().lower()
        if choice == 'y':
            known_words[word] = "Unacceptable"
            if word in word_vectors:
                del word_vectors[word]
            print(f"  [x] '{word}' is now permanently blacklisted.")
        else:
            known_words[word] = "Acceptable"
            print(f"  [✓] Kept '{word}' safe in active vocabulary.")
    suspected_stop_words = {}
    return word_vectors, known_words

def gobbler(paragraph_text, word_vectors, known_words):
    cleaned_chars = ""
    for char in paragraph_text.lower():
        if char.isalnum() or char.isspace():
            cleaned_chars += char
        else:
            cleaned_chars += " " 
    raw_tokens = cleaned_chars.split()
    clean_words = [word for word in raw_tokens if known_words.get(word) != "Unacceptable"]
    total_clean_count = len(clean_words)
    if total_clean_count == 0:
        return word_vectors, known_words
    for word in clean_words:
        if word not in word_vectors:
            word_vectors = vectorization(word, word_vectors)
            if word not in known_words:
                known_words[word] = "Acceptable"
    if total_clean_count <= 150:
        density_threshold = 0.025   # 2.5% for short entries (requires high repetition)
    elif total_clean_count <= 800:
        density_threshold = 0.015   # 1.5% for standard length blogs
    else:
        density_threshold = 0.005   # 0.5% for massive essays or Wikipedia pages
    local_counts = {}
    for word in clean_words:
        local_counts[word] = local_counts.get(word, 0) + 1
        
    local_densities = {}
    for word, count in local_counts.items():
        local_densities[word] = count / total_clean_count
    dominant_words = [
        word for word, density in local_densities.items() 
        if density >= density_threshold
    ]
    
    for i in range(len(dominant_words)):
        for j in range(i + 1, len(dominant_words)):
            w1 = dominant_words[i]
            w2 = dominant_words[j]
            word_vectors = converge_vectors(w1, w2, word_vectors)
            ledger_update(w1, w2)   
    return word_vectors, known_words

def grouper(word_vectors, similarity_threshold=70.0):
    words = list(word_vectors.keys())
    visited = set()
    relations_dictionary = {}
    group_counter = 1
    def get_similarity(vec_a, vec_b):
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(a * a for a in vec_a))
        mag_b = math.sqrt(sum(b * b for b in vec_b))
        if mag_a == 0 or mag_b == 0: 
            return 0.0
        return (dot_product / (mag_a * mag_b)) * 100
    for i, base_word in enumerate(words):
        if base_word in visited:
            continue
        pool = [base_word]
        for target_word in words:
            if base_word != target_word and target_word not in visited:
                score = get_similarity(word_vectors[base_word], word_vectors[target_word])
                if score >= similarity_threshold:
                    pool.append(target_word)
        if len(pool) > 1:
            group_dict = {}
            for member in pool:
                total_score = 0.0
                connections = 0
                for peer in pool:
                    if member == peer:
                        total_score += 100.0  # Perfect alignment with itself
                    else:
                        total_score += get_similarity(word_vectors[member], word_vectors[peer])
                    connections += 1
                # The confidence is the average fit within the group
                group_dict[member] = round(total_score / connections, 2)
                visited.add(member)
            group_key = f"r{group_counter}"
            relations_dictionary[group_key] = group_dict
            group_counter += 1
    return relations_dictionary

def main():
    print("=================================================================")
    print(" SEMANTIC STRUCTURAL MATRIX ENGINE ONLINE")
    print("=================================================================")
    print("Description:")
    print("This system analyzes textual density without relying on massive,")
    print("bloated relational databases. It dynamically streams learning updates")
    print("into 15-dimensional vectors, automatically flags structural stop-words")
    print("via dynamic connectivity scoring, and groups concept domains based on")
    print("round-robin confidence alignment tournaments.")
    print("=================================================================")
    word_vectors, known_words = load_brain_memory()
    while True:
        print("\n[MAIN MENU]")
        print("1 -> Ingest text blocks (Engage Gobbler)")
        print("2 -> Finalize, Run Diagnostics, Save and Exit")
        choice = input("Select an option (1-2): ").strip()
        if choice == '1':
            print("\nEnter or paste your paragraph text below (Type 'DONE' on a new line to process):")
            lines = []
            while True:
                line = input()
                if line.strip() == "DONE":
                    break
                lines.append(line)
            
            passage = " ".join(lines)
            if passage.strip():
                word_vectors, known_words = gobbler(passage, word_vectors, known_words)
            else:
                print("[-] Empty passage input canceled.")
        elif choice == '2':
            print("\nShutting down training loops... Initiating synchronization pipeline.")
            save_brain_memory(word_vectors, known_words)
            word_vectors, known_words = review_stop_words(word_vectors, known_words)
            save_brain_memory(word_vectors, known_words)
            print("\n[*] Processing vector alignments via round-robin cluster calculations...")
            final_relations = grouper(word_vectors)
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(final_relations, f, indent=4) 
            print("\n=======================================================")
            print("📊 COMPILED ARCHIVAL RELATION MAPPINGS:")
            print("=======================================================")
            print(json.dumps(final_relations, indent=4))
            print(f"\n[✓] Historical relation tracking catalog saved to '{HISTORY_FILE}'.")
            print("[!] Core execution successfully completed. Matrix offline.\n")
            break
        else:
            print("[-] Invalid execution choice. Select 1 or 2.")

if __name__ == "__main__":
    main()

