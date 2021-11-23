import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords

min_confidence = 0.5  # min confidence required to consider the object detected
debug = True
delimiters = r'[\s.\"\'-]+'
stop_words = set(stopwords.words('english'))
stemmer = nltk.stem.SnowballStemmer('english')
temporary_directory = 'temporary_directory'
max_retrieved_documents = 200

model_path='model_checkpoints/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
word_map_path='model_checkpoints/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
