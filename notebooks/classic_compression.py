# To download the data use `bash data/download_texts.sh`

import bz2
from collections import Counter
import heapq
import lzma
import zlib

from nltk import sent_tokenize
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_data(path):
    df = pd.read_csv(path, index_col=0)
    samples = df['text']
    texts = []
    for text_sample in samples:
        sentences = sent_tokenize(text_sample)
        suffix_text = ' '.join(sentences[len(sentences)//2:])
        texts.append(suffix_text)
    return texts

def huffman_code_for_frequencies(frequencies):
    """
Create a Huffman code dictionary {symbol: code} given symbol frequencies.
    """
    heap = []
    for symbol, freq in frequencies.items():
        # Each entry on the heap is (freq, unique_id, tree_node)
        # The 'unique_id' ensures no tie-breaking errors in Python's heap
        heap.append((freq, len(heap), symbol))
    heapq.heapify(heap)

    # Build Huffman tree
    while len(heap) > 1:
        freq1, _, left = heapq.heappop(heap)
        freq2, _, right = heapq.heappop(heap)
        new_node = (left, right)
        new_freq = freq1 + freq2
        heapq.heappush(heap, (new_freq, len(heap), new_node))

    # Only one element remains
    _, _, root = heap[0]

    # Traverse tree to get codes
    code_dict = {}

    def traverse(node, prefix):
        if isinstance(node, tuple):
            left, right = node
            traverse(left, prefix + "0")
            traverse(right, prefix + "1")
        else:
            code_dict[node] = prefix

    traverse(root, "")
    return code_dict

def huffman_encode(data, code_dict):
    """
    Encodes data using a Huffman code dictionary.
    Returns the bitstring or bit length.
    """
    encoded_bits = "".join(code_dict[symbol] for symbol in data)
    return encoded_bits


def example_huffman_compress():
    data = "this is a sample text to compress"
    freq = Counter(data)

    # Create Huffman codes from frequencies
    code_dict = huffman_code_for_frequencies(freq)

    # Encode
    encoded = huffman_encode(data, code_dict)

    # Calculate compression ratio (bits vs. original ASCII bits)
    original_bits = len(data) * 8
    compressed_bits = len(encoded)

    print("Original size (bits):", original_bits)
    print("Huffman-encoded size (bits):", compressed_bits)
    print("Compression ratio:", compressed_bits / original_bits)


if __name__ == '__main__':
    texts = get_data('pg19_valid_1k_chunks.csv')

    ratios_zlib = []
    ratios_bz2 = []
    ratios_lzma = []
    num_bits_list = []
    for s in tqdm(texts):
        data = s.encode('utf-8')
        num_bits_list.append(len(data) * 8)
        compressed_zlib = zlib.compress(data)
        ratio_zlib = len(compressed_zlib) / len(data)
        ratios_zlib.append(ratio_zlib)
        # bz2
        compressed_bz2 = bz2.compress(data)
        ratio_bz2 = len(compressed_bz2) / len(data)
        ratios_bz2.append(ratio_bz2)

        # lzma
        compressed_lzma = lzma.compress(data)
        ratio_lzma = len(compressed_lzma) / len(data)
        ratios_lzma.append(ratio_lzma)
    print('Lib: mean, std')
    print("zlib compression ratio:", f'{np.mean(ratios_zlib):.2f}, {np.std(ratios_zlib):.2f}')
    print("bz2 compression ratio:", f'{np.mean(ratios_bz2):.2f}, {np.std(ratios_bz2):.2f}')
    print("lzma compression ratio:", f'{np.mean(ratios_lzma):.2f}, {np.std(ratios_lzma):.2f}')
    print('Num bits:', f'{np.mean(num_bits_list):.2f}, {np.std(num_bits_list):.2f}')

    full_text = '\n'.join(texts).encode('utf-8')
    compressed_zlib = zlib.compress(full_text)
    ratio_zlib = len(compressed_zlib) / len(full_text)
    print("Full-text zlib compression ratio:", ratio_zlib)
    compressed_bz2 = bz2.compress(full_text)
    ratio_bz2 = len(compressed_bz2) / len(full_text)
    print("Full-text bz2 compression ratio:", ratio_bz2)

    # lzma
    compressed_lzma = lzma.compress(full_text)
    ratio_lzma = len(compressed_lzma) / len(full_text)
    print("Full-text lzma compression ratio:", ratio_lzma)
