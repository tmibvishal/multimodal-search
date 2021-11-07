from caption_decoder_encoder import get_caption_decoder_encoder
from captions_object_detection import get_captions_object_detection


def main():
    words2 = get_captions_object_detection(img_path='data/1000268201_693b08cb0e.jpg')
    print(words2)
    words = get_caption_decoder_encoder(img_path='data/1000268201_693b08cb0e.jpg',
                                        model_path='model_checkpoints/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar',
                                        word_map_path='model_checkpoints/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json')
    print(words)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
