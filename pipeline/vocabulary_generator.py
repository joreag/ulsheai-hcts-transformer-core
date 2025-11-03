import json
import os
import argparse

from tqdm import tqdm

class CharacterVocabularyGenerator:
    """
    Creates a simple, definitive character-level vocabulary based on a predefined,
    comprehensive character set. This is the foundational alphabet for JARVITS.
    """
    def __init__(self, project_root_path: str = None):
        if project_root_path is None:
            # Assumes src/ is one level down from project root
            project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.project_root = project_root_path
        
        # --- THE DEFINITIVE, EXPANDED CHARACTER SET ---
        self.special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        self.letters = list('abcdefghijklmnopqrstuvwxyz')
        self.numbers = list('0123456789')
        # Expanded symbols to cover math, logic, and file paths
        self.symbols = list(" .,=*/<>()[]+-%.':{}_\"\\|~`!@#$^&?\n;æɛʃ") 
        
        print("Character Vocabulary Generator initialized with expanded character set.")

    def generate_and_save(self, output_filename='vocab.json'):
        """Builds and saves the character-level vocabulary."""
        print("1. Assembling character-level vocabulary...")
        
        # Combine all characters, ensuring no duplicates and maintaining order
        # Using a dict to handle uniqueness automatically before creating the final map
        temp_charset = {}
        for char in self.special_tokens + self.letters + self.numbers + self.symbols:
            if char not in temp_charset:
                temp_charset[char] = True
        
        # Create the final dictionary mapping each character to a unique integer ID
        vocab_dict = {char: i for i, char in enumerate(temp_charset.keys())}

        print(f"2. Vocabulary created with {len(vocab_dict)} unique character tokens.")

        output_filepath = os.path.join(self.project_root, output_filename)
        print(f"3. Writing vocabulary to '{output_filepath}'...")
        with open(output_filepath, 'w') as f:
            json.dump(vocab_dict, f, indent=2)
        print("   -> Character vocabulary generation complete.")
        return vocab_dict

    def verify_against_curriculum(self, curriculum_path: str, vocab_dict: dict):
        """
        Scans the curriculum to ensure all used characters are in our vocabulary.
        """
        print(f"\n4. Verifying vocabulary against curriculum at '{curriculum_path}'...")
        missing_chars = set()
        
        filepaths = []
        for root, _, files in os.walk(curriculum_path):
            for filename in files:
                if filename.endswith(".json"):
                    filepaths.append(os.path.join(root, filename))

        for filepath in tqdm(filepaths, desc="Verifying Characters"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                for char in content.lower():
                    if char not in vocab_dict:
                        missing_chars.add(char)
            except Exception as e:
                print(f"   -> WARNING: Could not read '{filepath}': {e}")
        
        if not missing_chars:
            print("   -> SUCCESS: Vocabulary is comprehensive. All characters in curriculum are covered.")
        else:
            print(f"   -> WARNING: Found {len(missing_chars)} characters in curriculum not present in vocabulary!")
            print(f"      Missing characters: {sorted(list(missing_chars))}")
            print("      Consider adding them to the 'symbols' list in this script.")
        print("Verification complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and verify the JARVITS character-level vocabulary.")
    parser.add_argument('--output', type=str, default='vocab_genesis.json', help="Output filename for the vocabulary.")
    parser.add_argument('--curriculum-path', type=str, default='curriculum/', help="Path to the curriculum directory for verification.")
    args = parser.parse_args()

    print("--- Running JARVITS Character-Level Vocabulary Generation ---")
    generator = CharacterVocabularyGenerator()
    new_vocab = generator.generate_and_save(output_filename=args.output)
    
    # After generating, run the verification step
    if args.curriculum_path and os.path.isdir(args.curriculum_path):
        generator.verify_against_curriculum(args.curriculum_path, new_vocab)