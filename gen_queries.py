import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from caption_decoder_encoder import get_caption_decoder_encoder
from caption_object_detection import get_captions_object_detection
from config import temporary_directory, max_retrieved_documents, model_path, word_map_path


def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-i', '--image_caption_txt_path', type=str, required=True,
                        help="Path for the image caption txt file")
    parser.add_argument('-q', '--query_file', type=str, required=True,
                        help="Path for the pickle file of queries generated")
    args = vars(parser.parse_args())
    return args


def save_captions_collections_from_image_captions(image_caption_file_path, captions_collection_path):
    """removes the column image and saves the csv without header"""
    df = pd.read_csv(image_caption_file_path, delimiter=',')
    df.drop('image', inplace=True, axis=1)
    df.to_csv(captions_collection_path, header=False, index=False)



def gen_query(image_path):
    start = time.time()
    words = get_caption_decoder_encoder(img_path=image_path)
    words2 = get_captions_object_detection(img_path=image_path)
    end = time.time()
    print("Time taken to generate queries", end-start)

    if len(words) >= 2:
        lstmquery = ' '.join(words[1:-1])  # removing <start> and <end> from words
    else:
        lstmquery = ''
    odquery = ' '.join(words2)
    print(f'query generated: {lstmquery}')
    print(f'query generated: {odquery}')
    return lstmquery, odquery

def main(args):
    image_caption_txt_path = args['image_caption_txt_path']
    if not os.path.exists(temporary_directory):
        os.makedirs(temporary_directory)

    captions_collection_path = os.path.join(temporary_directory, os.path.basename(image_caption_txt_path))
    save_captions_collections_from_image_captions(image_caption_txt_path, captions_collection_path)
    output_dir_path = os.path.join(temporary_directory, 'output')
    df = pd.read_csv(image_caption_txt_path)
    images_directory = os.path.join(os.path.dirname(image_caption_txt_path), 'Images')
    queries_path = args['query_file']
    queries = {}

    for image_name in df['image'].unique():
        image_path = os.path.join(images_directory, image_name)
        queries[image_name] = gen_query(image_path)
    with open(queries_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = parse_args()
    main(args)