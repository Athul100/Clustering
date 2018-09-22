import json
from pre_process import pre_process
from model import create_vector, k_means
import numpy as np
import matplotlib.pyplot as plt
import pickle
from word_vec import Word2Vec, SpellCorrect
from word_segementation import WordSegment

with open('data.json',  encoding="utf8") as f:
    data = json.load(f)

word_vec = Word2Vec()
spell_correct = SpellCorrect(word_vec.model)
word_segment = WordSegment()
tweets = []


for each_data in data:
    content = each_data.get('content')
    content = pre_process(content, spell_correct, word_segment)
    if tweets.__contains__(content) or content is None:
        pass
    else:
        tweets.append(str(content))
try:
    with open('tweets.pkl', 'wb') as output:
        pickle.dump(tweets, output, pickle.HIGHEST_PROTOCOL)
except:
    pass
try:
    with open('file.txt', 'w') as f:
        for item in tweets:
            f.write("%s\n" % item)
except:
    pass

print('write file completed')
tfs = create_vector(tweets)
print(tfs)
k = 50
km = k_means(tfs, k)

new_json = {}

for i in set(km.labels_):
    current_cluster_bills = [tweets[x] for x in np.where(km.labels_ == i)[0]]
    new_json['class_' + str(i)] = current_cluster_bills

with open('out.json', 'w') as outfile:
    json.dump(new_json, outfile)

plt.hist(km.labels_, bins=k)
plt.show()




