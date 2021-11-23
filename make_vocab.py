import argparse
import os
import pickle
from pathlib import Path
import collections
import numpy as np
import pandas as pd

# from caption_decoder_encoder import get_caption_decoder_encoder
# from caption_object_detection import get_captions_object_detection
from config import temporary_directory, max_retrieved_documents
from search_image import main as search_img
from search_image import get_top_docs, make_collection, bm2_score, lm_score, vsm_score, lmt_score
from utils import splitter, tokenize_and_store
from evaluation import save_captions_collections_from_image_captions


captions_collection_path = os.path.join(temporary_directory, os.path.basename('data/flickr8k/image_caption.txt'))
print(captions_collection_path)
save_captions_collections_from_image_captions('data/flickr8k/image_caption.txt', captions_collection_path)

vocab = set()

with open(captions_collection_path, 'r') as f:
	for line in f:
		dictionary = collections.defaultdict(int)
		tokenize_and_store(line, dictionary, frequency=True, raw=True)
		# print(dictionary)
		# break
		for word in dictionary:
			vocab.add(word)


with open("temporary_directory/collection_vocab.pkl", 'wb') as f:
    pickle.dump(vocab, f)