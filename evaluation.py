import argparse
import os
import pickle
import timeit

import numpy as np
import pandas as pd

from config import temporary_directory, max_retrieved_documents
from search_image import get_top_docs, make_collection, bm2_score, lm_score, vsm_score, lmt_score


# -i data/flickr8k/image_caption.txt -c data/collection_saved.pkl -q data/queries_8k.pkl -o output/output_bm25 -f bm25

def parse_args():
    parser = argparse.ArgumentParser(description='Get evaluation for model')
    parser.add_argument('-i', '--image_caption_txt_path', type=str, required=True,
                        help='Path for the image caption txt file')
    parser.add_argument('-c', '--collection_file', type=str, required=True,
                        help='Path for the pickle file of collection')
    parser.add_argument('-q', '--query_file', type=str, required=True,
                        help='Path for the pickle file of queries generated')
    parser.add_argument('-o', '--output_file_path', type=str, required=True,
                        help='Output file path')
    parser.add_argument('-f', '--scoring_function', type=str, required=True,
                        choices=['vsm', 'bm25', 'lm', 'lmt'],
                        help='Scoring Function vsm/bm25/lm/lmt')
    args_ = vars(parser.parse_args())
    return args_


def save_captions_collections_from_image_captions(image_caption_file_path, captions_collection_path):
    """removes the column image and saves the csv without header"""
    df = pd.read_csv(image_caption_file_path, delimiter=',')
    df.drop('image', inplace=True, axis=1)
    df.to_csv(captions_collection_path, header=False, index=False)


def evaluate(image_caption_df, image_name, top_k, output_file=None):
    result_string = ''

    # Checking some assertions
    temp = [x[0] for x in top_k]
    assert temp == sorted(temp, reverse=True)

    retrieved_documents = [x[1] for x in top_k]
    assert len(retrieved_documents) <= max_retrieved_documents

    relevant_documents_list = list(image_caption_df.query(f'image == "{image_name}"')['caption'])
    relevant_documents = set(relevant_documents_list)

    # print(relevant_documents)
    # print(retrieved_documents)

    if output_file is not None:
        output_file.write(f'relevant_documents: {relevant_documents}\n')
        output_file.write(f'retrieved_documents: {retrieved_documents}\n')

    print(f'relevant query 1: {relevant_documents_list[0]}')
    print(retrieved_documents)

    DCG = np.zeros(max(51, len(retrieved_documents)), dtype=np.float)
    AP = 0
    precision_numerator = 0
    rec_at_1, rec_at_5, rec_at_10, rec_at_100 = 0, 0, 0, 0
    for i, document_text in enumerate(retrieved_documents):

        rel = 1 if document_text in relevant_documents else 0
        if i < 1:
            rec_at_1 += rel
        if i < 5:
            rec_at_5 += rel
        if i < 10:
            rec_at_10 += rel
        if i < 100:
            rec_at_100 += rel
        DCG[i] = rel / np.log2(i + 2)
        if i > 0:
            DCG[i] += DCG[i - 1]
        precision_numerator += rel
        precision = precision_numerator / (i + 1)
        AP += precision
    AP = AP / 100

    # Fill remaining DCGs
    for i in range(len(retrieved_documents), max(len(retrieved_documents), 51)):
        if i > 0:
            DCG[i] += DCG[i - 1]

    # Calculating DCG'
    DCG_prime = np.zeros(max(len(retrieved_documents), 51), dtype=np.float)
    for i in range(max(len(retrieved_documents), 51)):
        rel = 1 if i < 5 else 0
        DCG_prime[i] = rel / np.log2(i + 2)
        if i > 0:
            DCG_prime[i] += DCG_prime[i - 1]
    nDCG = DCG / DCG_prime

    rec_at_1 /= 5
    rec_at_5 /= 5
    rec_at_10 /= 5
    rec_at_100 /= 5

    result_string += f'query_image = {image_name}\n'
    result_string += f'nDCG[5] = {nDCG[5]} and nDCG[10] = {nDCG[10]} and nDCG[50] = {nDCG[50]}\n'
    result_string += f'Average Precision (MAP) = {AP}\n'
    result_string += f'Recall at k:\n'
    result_string += f'r@1 {rec_at_1}\n'
    result_string += f'r@5 {rec_at_5}\n'
    result_string += f'r@10 {rec_at_10}\n'
    result_string += f'r@100 {rec_at_100}\n'
    result_string += '-------------------------------\n'

    return result_string, np.array([AP, rec_at_1, rec_at_5, rec_at_10, rec_at_100, nDCG[5], nDCG[10], nDCG[50]], dtype=np.float64)


def create_collection_file(caption_collection_txt, collection_saved, trans_probs=None):
    save = False
    if not os.path.isfile(collection_saved):
        collection = make_collection(caption_collection_txt, trans_probs)
        if not os.path.exists(os.path.dirname(collection_saved)):
            os.makedirs(os.path.dirname(collection_saved))
        save = True
        # with open(collection_saved, 'wb') as handle:
        #     pickle.dump(collection, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(collection_saved, 'rb') as handle:
            collection = pickle.load(handle)
    return collection, save


# Assume queries have already been generated before calling evaluation.py
def main(args):
    image_caption_txt_path = args['image_caption_txt_path']
    images_directory = os.path.join(os.path.dirname(image_caption_txt_path), 'Images')
    if not os.path.exists(temporary_directory):
        os.makedirs(temporary_directory)
    df = pd.read_csv(image_caption_txt_path)

    scoring_function_name = args['scoring_function']
    if scoring_function_name == 'vsm':
        scoring_function = vsm_score
    elif scoring_function_name == 'bm25':
        scoring_function = bm2_score
    elif scoring_function_name == 'lm':
        scoring_function = lm_score
    elif scoring_function_name == 'lmt':
        scoring_function = lmt_score
    else:
        print('error: Scoring function hasn\'t been implemented')
        exit(-1)

    raw = (scoring_function_name == 'lmt')

    with open(args['query_file'], 'rb') as f:
        queries = pickle.load(f)

    trans_probs = None
    with open('data/word2vec/translation_probs.pkl', 'rb') as f:
        trans_probs = pickle.load(f)

    captions_collection_path = os.path.join(temporary_directory, os.path.basename(image_caption_txt_path))
    save_captions_collections_from_image_captions(image_caption_txt_path, captions_collection_path)
    collection, save_collection = create_collection_file(captions_collection_path, args['collection_file'], trans_probs)

    output_file = open(args['output_file_path'], 'w')
    i = 0
    results = np.zeros(8, dtype=np.float64)

    avg_time_taken = 0
    for image_name in df['image'].unique():

        start = timeit.default_timer()
        image_path = os.path.join(images_directory, image_name)
        query = queries[image_name][0]

        top_k = get_top_docs(query, collection, scoring_function, raw)
        output_file.write(f'image: {image_name}\n')
        output_file.write(f'query: {query}\n')
        result_string, cur_result = evaluate(df, image_name, top_k, output_file=output_file)
        output_file.write(result_string + '\n')
        results += cur_result
        time_taken = timeit.default_timer() - start
        output_file.write(f'Time: {time_taken} sec\n')
        avg_time_taken += time_taken
        print(f'query_number {i}, time_taken {time_taken} sec\n')

        i += 1

        # if i == 1000:
        #     break

    # AP, rec_at_1, rec_at_5, rec_at_10, rec_at_100, nDCG[5], nDCG[10], nDCG[50]
    avg_time_taken /= i
    results /= i
    output_file.write(
        f'AP: {results[0]}, rec_at_1: {results[1]}, rec_at_5: {results[2]}, rec_at_10: {results[3]}, rec_at_100: {results[4]}, ' +
        f'nDCG[5]: {results[5]}, nDCG[10]: {results[6]}, nDCG[50]: {results[7]}\n')
    output_file.write(f'avg_time_taken_per_query: {avg_time_taken}')
    output_file.close()

    if save_collection:
        with open(args['collection_file'], 'wb') as handle:
            pickle.dump(collection, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main(parse_args())
