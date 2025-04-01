from typing import List

from nltk import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm


device = torch.device('cuda')


def compress_with_llm(text: str, model, tokenizer) -> bytes:
    """Compress text using arithmetic coding with LLM probabilities in a single inference."""
    input_ids = tokenizer(text, return_tensors='pt')['input_ids']
    
    # Get all probabilities at once with a single model inference
    with torch.no_grad():
        outputs = model(input_ids, return_dict=True)
        # Shape: [batch_size, seq_len, vocab_size]
        all_logits = outputs.logits
        
        # Convert logits to probabilities for each position
        # We need position i to predict token i+1
        all_probs = []
        for i in range(input_ids.shape[1] - 1):  # -1 because last position has no next token
            probs = torch.softmax(all_logits[0, i, :], dim=-1).cpu().numpy()
            all_probs.append(probs)
    
    # Initialize arithmetic coding state
    low = 0
    high = 0xFFFFFFFF
    pending_bits = 0
    compressed_bits = []
    
    # Process each token using the pre-computed probabilities
    for token_idx in range(input_ids.shape[1] - 1):
        # Get next token (the one we're predicting)
        next_token_id = input_ids[0, token_idx + 1].item()
        probs = all_probs[token_idx]
        
        # Calculate cumulative probability distribution
        cum_probs = np.zeros(len(probs) + 1)
        for j in range(len(probs)):
            cum_probs[j+1] = cum_probs[j] + probs[j]
        cum_probs[-1] = 1.0  # Ensure the last value is exactly 1.0
        
        # Update interval based on token probability
        range_size = high - low + 1
        high_new = low + int(range_size * cum_probs[next_token_id + 1]) - 1
        low_new = low + int(range_size * cum_probs[next_token_id])
        low, high = low_new, high_new
        
        # Emit bits and rescale when needed
        while True:
            if high < 0x80000000:  # MSB = 0: both in lower half
                # Emit 0 followed by pending 1's
                compressed_bits.append(0)
                compressed_bits.extend([1] * pending_bits)
                pending_bits = 0
                # Scale up interval: [0, 0.5) → [0, 1)
                low <<= 1
                high = (high << 1) | 1
                
            elif low >= 0x80000000:  # MSB = 1: both in upper half
                # Emit 1 followed by pending 0's
                compressed_bits.append(1)
                compressed_bits.extend([0] * pending_bits)
                pending_bits = 0
                # Scale up interval: [0.5, 1) → [0, 1)
                low = (low << 1) & 0xFFFFFFFF
                high = ((high << 1) & 0xFFFFFFFF) | 1
                
            elif low >= 0x40000000 and high < 0xC0000000:  # Straddles middle
                # Underflow case: interval in [0.25, 0.75)
                pending_bits += 1
                # Scale up interval: [0.25, 0.75) → [0, 1)
                low = (low << 1) & 0x7FFFFFFF
                high = ((high << 1) & 0x7FFFFFFF) | 0x80000001
                
            else:
                # No more scaling possible
                break
    
    # Finalize by outputting a value in the final range
    if low < 0x40000000:
        compressed_bits.append(0)
        compressed_bits.extend([1] * (pending_bits + 1))
    else:
        compressed_bits.append(1)
        compressed_bits.extend([0] * (pending_bits + 1))
    
    # Convert bits to bytes
    result_bytes = bytearray()
    for i in range(0, len(compressed_bits), 8):
        byte = 0
        for j in range(min(8, len(compressed_bits) - i)):
            byte |= compressed_bits[i + j] << (7 - j)
        result_bytes.append(byte)
    
    return bytes(result_bytes)

def get_random_text(max_length=1568):
    vocab = []
    with open('./data/vocab_100k.txt') as fin:
        for line in fin:
            vocab += [line.strip()]
    cur_max_length = np.random.randint(2, max_length+1)
    return ' '.join(np.random.choice(vocab, size=cur_max_length * 5))


def get_data(path, max_text_length):
    if path == 'random':
        n_docs = 1000
        return [get_random_text(max_text_length) for _ in range(n_docs)]

    path = f'./data/{path}'
    df = pd.read_csv(path, index_col=0)
    samples = df['text']
    texts = []
    for text_sample in samples:
        sentences = sent_tokenize(text_sample)
        suffix_text = ' '.join(sentences[len(sentences)//2:])
        # take roughly max_text_length words
        suffix_text = ' '.join(word_tokenize(suffix_text)[:max_text_length])
        texts.append(suffix_text)
    return texts


if __name__ == '__main__':
    model_list = [
        'EleutherAI/pythia-160m',
        'EleutherAI/pythia-2.8b',
        'facebook/opt-1.3b',
        'princeton-nlp/Sheared-LLaMA-1.3B',
        'meta-llama/Llama-3.2-1B'
    ]
    model_name = model_list[0]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    max_text_length = 1568
    print(f'max_text_length: {max_text_length} (words counted with nltk.word_tokenize)')
    for dataset in ['pg19_valid_1k_chunks.csv', 'fanfics_1k_chunks.csv', 'random']:
        print(f'=== dataset: {dataset} ===')
        texts = get_data(dataset, max_text_length)

        ratios = []
        num_bits_list = []
        for s in tqdm(texts):
            data = s.encode('utf-8')
            num_bits_list.append(len(data) * 8)
            compressed = compress_with_llm(s, model, tokenizer)
            ratio = len(data) / len(compressed)
            ratios.append(ratio)
        
        print('Model: mean+-std')
        print("{model_name} compression ratio:", f'{np.mean(ratios_zlib):.2f}+-{np.std(ratios_zlib):.2f}')


   # [0, 1)
# <---------->
# [0, 0.6]  [0.6, 1) 
      # <---->
# <-----> 
# [0, 0.1) [0.1, 0.6)
  # <--->
# <->

