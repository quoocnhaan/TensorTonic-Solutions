import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        self.word_to_id = {}
        self.id_to_word = {}
    
        vocab = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        seen = set(vocab)
        
        for sentence in texts:
            for word in sentence.lower().split():
                if word not in seen:
                    vocab.append(word)
                    seen.add(word)
    
        self.vocab_size = len(vocab)
    
        for i, word in enumerate(vocab):
            self.word_to_id[word] = i
            self.id_to_word[i] = word

            
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        ids = []
        
        for word in text.lower().split():
            # Use get() with the ID of the UNK token as the fallback
            ids.append(self.word_to_id.get(word,self.word_to_id.get(self.unk_token, 1)))
                
        return ids
    
    def decode(self, ids: List[int]) -> str:
        words = []
    
        for i in ids:
            # Safely check against eos_token ID
            if i == self.word_to_id.get(self.eos_token):
                break
    
            # Safely skip pad_token and bos_token IDs
            if i in (
                self.word_to_id.get(self.pad_token),
                self.word_to_id.get(self.bos_token),
            ):
                continue
    
            # Append the word, fallback to the string "<UNK>" if ID is completely unknown
            words.append(self.id_to_word.get(i, self.unk_token))
    
        return " ".join(words)