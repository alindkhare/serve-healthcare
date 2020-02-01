import pickle

with open('res.pkl','rb') as fin:
    res = pickle.load(fin)
print(res)