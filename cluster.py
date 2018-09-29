import json
from pre_process import pre_process
from model import create_vector, k_means, create_combined_vector
import numpy as np
#import matplotlib.pyplot as plt
import pickle
from word_vec import Word2Vec, SpellCorrect
from word_segementation import WordSegment

with open('data.json',  encoding="utf8") as f:
    data = json.load(f)

word_vec = Word2Vec()
spell_correct = SpellCorrect(word_vec.model)
word_segment = WordSegment()
tweets = []
orginal_content = []

try:
    with open('tweets.pkl', 'rb') as output:
        tweets  = pickle.load(output)
except:
    pass

try:
    with open('org_tweets.pkl', 'rb') as output:
        orginal_content  = pickle.load(output)
except:
    pass

if len(tweets) < 1 :
    for each_data in data:
        content = each_data.get('content')
        processed_content = pre_process(content, spell_correct, word_segment)
        if tweets.__contains__(processed_content) or processed_content is None:
            pass
        else:
            tweets.append(str(processed_content))
            orginal_content.append(str(content))
    try:
        with open('tweets.pkl', 'wb') as output:
            pickle.dump(tweets, output, pickle.HIGHEST_PROTOCOL)
    except:
        pass
    try:
        with open('org_tweets.pkl', 'wb') as output:
            pickle.dump(orginal_content, output, pickle.HIGHEST_PROTOCOL)
    except:
        pass


    print('write file completed')
# tfs = create_combined_vector(tweets, word_vec.model)
tfs = create_vector(tweets)
print(np.where(np.isnan(tfs )))
k = 50
km = k_means(tfs, k)

new_json = {}

for i in set(km.labels_):
    current_cluster_bills = [orginal_content[x] for x in np.where(km.labels_ == i)[0]]
    new_json['class_' + str(i)] = current_cluster_bills



with open('out.json', 'w') as outfile:
    json.dump(new_json, outfile)

#plt.hist(km.labels_, bins=k)
#plt.show()




