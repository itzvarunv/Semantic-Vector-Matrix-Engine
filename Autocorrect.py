import json
import os
import jellyfish
import math

MEMORY_FILE = "memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                data = json.load(f)
                return data.get("learned_vectors", {}), data.get("saved_hashes", {})
        except json.JSONDecodeError:
            print("Memory file was corrupted. Starting with a blank slate.")
            return {}, {}
    else:
        print("No existing memory found. Starting a fresh database.")
        return {}, {}

def save_memory(learned_vectors, saved_hashes):
    data_to_save = {
        "learned_vectors": learned_vectors,
        "saved_hashes": saved_hashes
    }
    with open(MEMORY_FILE, "w") as f:
        json.dump(data_to_save, f, indent=4)
    print("Progress saved")

def get_phonetic_alphabet_value(char):
    char = char.lower()
    if char in 'aeiouy': return 5       # Vowels / Glides
    elif char in 'bpvf': return 10      # Labials / Friction sounds
    elif char in 'cszx': return 15      # Sibilants
    elif char in 'dtlnmr': return 20    # Liquids / Dentals
    elif char in 'gkqj': return 25      # Gutturals
    return 2                            # Default for rare sounds (H, W)

def learner(learned_vectors, saved_hashes):
    correct_word = input("\nEnter a guaranteed correct word to learn: ")
    cleaned_word = correct_word.strip().lower()
    if not cleaned_word:
        print("Invalid empty input.")
        return
    if cleaned_word in learned_vectors:
        print(f" Notice: '{cleaned_word}' is already in memory! Skipping.")
        return 
    word_len = len(cleaned_word)
    group_dictionary = {}
    for i in range(1, word_len + 1):
        char = cleaned_word[i - 1]
        group_dictionary[f"a{i}"] = get_phonetic_alphabet_value(char)
    word_hash = jellyfish.soundex(cleaned_word)
    if word_hash not in saved_hashes:
        saved_hashes[word_hash] = []
    saved_hashes[word_hash].append(cleaned_word)
    learned_vectors[cleaned_word] = group_dictionary
    print(f" Success! Learned: '{cleaned_word}' (Added to Hash Bucket: {word_hash})")
    save_memory(learned_vectors, saved_hashes) 
    
def calculate_vector_magnitude(vector):
    return math.sqrt(sum(val ** 2 for val in vector.values()))

def calculate_dot_product(vec_a, vec_b):
    all_keys = set(vec_a.keys()).union(set(vec_b.keys()))
    dot_product = 0.0
    for key in all_keys:
        val_a = vec_a.get(key, 0)
        val_b = vec_b.get(key, 0)
        dot_product += val_a * val_b   
    mag_a = calculate_vector_magnitude(vec_a)
    mag_b = calculate_vector_magnitude(vec_b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    similarity = dot_product / (mag_a * mag_b)
    return similarity * 100

def tester(learned_vectors, saved_hashes):
    unknown_word = input("\nEnter a word to test/check: ")
    test_word = unknown_word.strip().lower()
    if not test_word:
        print("Invalid input.")
        return
    test_vector = {}
    for i in range(1, len(test_word) + 1):
        char = test_word[i - 1]
        test_vector[f"a{i}"] = get_phonetic_alphabet_value(char)  
    test_hash = jellyfish.soundex(test_word)
    print(f"-> Testing Word: '{test_word}' (Hash Bucket: {test_hash})")
    close_relatives = saved_hashes.get(test_hash, [])     
    if not close_relatives:
        print(" Result: No phonetic relatives found in memory. Too jumbled to operate!")
        return
    print(f"-> Found phonetic candidates in this bucket: {close_relatives}")
    best_match_word = None
    highest_score = 0.0
    for relative in close_relatives:
        learned_vector = learned_vectors[relative]
        score = calculate_dot_product(learned_vector, test_vector)
        print(f"   * Match score with '{relative}': {score:.2f}%")
        if score > highest_score:
            highest_score = score
            best_match_word = relative
    if highest_score >= 80.0:
        print(f" Target Found! Did you mean: '{best_match_word}'? ({highest_score:.2f}% Match)")
    else:
        print(f" Result: Highest match was only {highest_score:.2f}%. Too jumbled to operate!")

if __name__ == "__main__":
    main_vectors, main_hashes = load_memory()
    while True:
        print("\n[AI Menu] 1: Learn Word | 2: Test Typo | 3: Exit")
        choice = input("Choose option: ").strip()
        if choice == '1':
            learner(main_vectors, main_hashes)
        elif choice == '2':
            tester(main_vectors, main_hashes)
        elif choice == '3':
            print("Shutting down AI.")
            break
        else:
            print("Invalid input.")
