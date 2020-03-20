import pickle

with open('../data/unique_subjects.pkl', 'rb') as inp:
    unique_exprs = pickle.load(inp)

print(unique_exprs[:10])