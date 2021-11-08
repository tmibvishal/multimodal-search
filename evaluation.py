import argparse
import os
import pickle
import timeit
from pathlib import Path

import numpy as np
import pandas as pd

from config import temporary_directory, max_retrieved_documents
from search_image import main as search_img


def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-i', '--image_caption_txt_path', type=str, required=True,
                        help="Path for the image caption txt file")
    parser.add_argument('-o', '--output_file_path', type=str, required=True,
                        help="Output file path")
    args = vars(parser.parse_args())
    return args


def save_captions_collections_from_image_captions(image_caption_file_path, captions_collection_path):
    """removes the column image and saves the csv without header"""
    df = pd.read_csv(image_caption_file_path, delimiter=',')
    df.drop('image', inplace=True, axis=1)
    df.to_csv(captions_collection_path, header=False, index=False)


def evaluate(image_caption_df, image_name, out_file_path):
    output_file = pd.read_csv(out_file_path, delimiter=',')
    result_string = ""

    # Checking some assertions
    temp = list(output_file['score'])
    assert temp == sorted(temp, reverse=True)

    retrieved_documents = output_file['document_text']
    assert len(retrieved_documents) <= max_retrieved_documents

    relevant_documents_list = list(image_caption_df.query(f'image == "{image_name}"')['caption'])
    relevant_documents = set(relevant_documents_list)
    print(f'relevant query 1: {relevant_documents_list[0]}')
    DCG = np.zeros(len(retrieved_documents), dtype=np.float)
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

    # Calculating DCG'
    DCG_prime = np.zeros(len(retrieved_documents), dtype=np.float)
    for i, _ in enumerate(retrieved_documents):
        rel = 1 if i < 5 else 0
        DCG_prime[i] = rel / np.log2(i + 2)
        if i > 0:
            DCG_prime[i] += DCG_prime[i - 1]
    nDCG = DCG / DCG_prime
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


def main(args):
    image_caption_txt_path = args['image_caption_txt_path']
    if not os.path.exists(temporary_directory):
        os.makedirs(temporary_directory)
    captions_collection_path = os.path.join(temporary_directory, os.path.basename(image_caption_txt_path))
    save_captions_collections_from_image_captions(image_caption_txt_path, captions_collection_path)
    collection_saved_path = os.path.join(temporary_directory, 'collection_saved.pkl')
    output_dir_path = os.path.join(temporary_directory, 'output')

    df = pd.read_csv(image_caption_txt_path)
    images_directory = os.path.join(os.path.dirname(image_caption_txt_path), 'Images')
    output_file = open(args['output_file_path'], "w")
    i = 0
    results = np.zeros(8, dtype=np.float64)
    queries_path = os.path.join(temporary_directory, 'queries.pkl')
    queries = {}
    # if not os.path.isfile(queries_path):
    #     if not os.path.exists(temporary_directory):
    #         os.makedirs(temporary_directory)
    #     with open(queries_path, 'rb') as handle:
    #         queries = pickle.load(handle)
    for image_name in df['image'].unique():
        print(i)
        image_path = os.path.join(images_directory, image_name)
        out_file_name = Path(image_path).stem + '.txt'
        output_file_path = os.path.join(output_dir_path, out_file_name)
        # --inp_path data/flickr8k/Images/667626_18933d713e.jpg --collection_saved data/collection_saved.pkl
        # --caption_collection_txt data/flickr8k/small_captions_collections.txt --output_file_path output/output1.txt
        args2 = {'inp_path': image_path, 'collection_saved': collection_saved_path, 'caption_collection_txt': captions_collection_path,
                 'output_file_path': output_file_path, 'score_method': 'bm25'}
        start = timeit.default_timer()
        query = search_img(args2)
        queries[image_path] = query
        output_file.write(f'image: {image_path}\n')
        output_file.write(f'query: {query}\n')
        result_string, cur_result = evaluate(df, image_name, output_file_path)
        output_file.write(result_string + '\n')
        results += cur_result
        output_file.write(f'Time: {timeit.default_timer() - start} sec\n')
        i += 1
    # AP, rec_at_1, rec_at_5, rec_at_10, rec_at_100, nDCG[5], nDCG[10], nDCG[50]
    results /= i
    output_file.write(f'AP: {results[0]}, rec_at_1: {results[1]}, rec_at_5: {results[2]}, rec_at_10: {results[3]}, rec_at_100: {results[4]}, nDCG[5]: {results[5]}, nDCG[10]: {results[6]}, nDCG[50]: {results[7]}\n')
    output_file.close()


    with open(queries_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = parse_args()
    main(args)
