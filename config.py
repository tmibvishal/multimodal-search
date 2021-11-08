import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords

min_confidence = 0.5  # min confidence required to consider the object detected
debug = True
delimiters = r'[.\s]\s*'
stop_words = set(stopwords.words('english'))
stemmer = nltk.stem.SnowballStemmer('english')
temporary_directory = 'temporary_directory'
max_retrieved_documents = 200
