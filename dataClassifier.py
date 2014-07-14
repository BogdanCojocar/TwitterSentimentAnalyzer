from nltk.corpus import wordnet
from nltk.corpus import stopwords
import nltk
           
def filter_data(old_data, new_data):
    # take only words in English
    for data in old_data:
        filtered_data = [data_word for data_word in data if wordnet.synsets(data_word)]
        filtered_data = [data_word for data_word in filtered_data if data_word not in stopwords.words('english')]
        if len(filtered_data) > 0:
            new_data.append(filtered_data)
            
def feature_extractor(data):
    data_words = set(data)
    features = {}
    for word in data_words:
        features['contains(%s)' % word] = (word in data_words)
    return features


def read_training_data():
    final_training_data = []
    pos_file = open('positives.txt', 'r')
    pos_data = pos_file.readlines()
    neg_file = open('negatives.txt', 'r')
    neg_data = neg_file.readlines()
    
    neg_tag = []
    pos_tag = []
    
    for _ in range(0, len(neg_data)):
        neg_tag.append('negative')
 
    for _ in range(0,len(pos_data)):
        pos_tag.append('positive')
 
    pos_training_data = zip(pos_data, pos_tag)
    neg_training_data = zip(neg_data, neg_tag)
 
    training_data = pos_training_data + neg_training_data
    
    for (data, label) in training_data:
        final_data = [word.lower() for word in data.split() if len(word) >=3]
        final_training_data.append((final_data, label))
        
    return final_training_data
        
def train(trainig_data):
    training_set = nltk.classify.apply_features(feature_extractor, trainig_data)
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    return classifier

