import pickle

with open("queries_8k.pkl", 'rb') as f:
	queries = pickle.load(f)

print(queries)
print(len(queries))