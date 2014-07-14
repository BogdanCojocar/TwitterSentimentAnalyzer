from twitterData import get_twitter_data
from dataClassifier import filter_data
from dataClassifier import read_training_data
from dataClassifier import train
from dataClassifier import feature_extractor
import os
import numpy as np
import matplotlib.pyplot as plt

def menu():
    print 'Menu'
    print '1. Show collected tweets'
    print '2. Show cleaned tweets'
    print '3. Show the classifier structure after training'
    print '4. Enter a sentence to check for sentiment'
    print '5. Show overall sentiment for the collected tweets.'
    print '6. Show graph with the sentiment distribution'
    print '7. Exit'
    
def clear() :
    os.system('cls')
        
def print_data(data):
    for element in data:
        print element
    
def main_program():
    option = '0'
    tweet_samples, filtered_samples, classifier = init()
    
    while option != '7': 
        menu()
        option = raw_input('Enter new option: ')
        
        if option == '1':
            print_data(tweet_samples)
        elif option == '2':
            print_data(filtered_samples)
        elif option == '3':
            print classifier.show_most_informative_features(n=30)
        elif option == '4':
            sentence = raw_input("Sentence: ")
            cleaned_sentence = [word.lower() for word in sentence.split() if len(word) >=3]
            print 'This sentence is ' + classifier.classify(feature_extractor(cleaned_sentence))
        elif option == '5':
            negative, positive = find_overall_sentiment(filtered_samples, classifier)
            print_sentiment(positive, negative)
        elif option == '6':
            plot_bar(filtered_samples, classifier)
        clear()
        
def init():
    print 'loading...'
    tweet_samples = get_twitter_data()
    filtered_tweet_samples = []
    filter_data(tweet_samples, filtered_tweet_samples)
    
    training_data = read_training_data()
    classifier = train(training_data)
    
    return tweet_samples, filtered_tweet_samples, classifier

def find_overall_sentiment(samples, classifier):
    positive = 0
    negative = 0
    for sample in samples:
        if classifier.classify(feature_extractor(sample)) == 'positive':
            positive += 1
        else:
            negative += 1
    
    return positive, negative

def print_sentiment(positive, negative):
    if positive > negative:
        print 'positive'
    else:
        print 'negative'
    
def plot_bar(samples, classifier):
    sentiment = ('Positive', 'Negative')
    y_pos = np.arange(len(sentiment))
    performance = find_overall_sentiment(samples, classifier)
    error = np.random.rand(len(sentiment))

    plt.barh(y_pos, performance, xerr=error, align='center', alpha=0.4)
    plt.yticks(y_pos, sentiment)
    plt.xlabel('Number of tweets')
    plt.title('Sentiment distribution')

    plt.show()

if __name__ == '__main__':
    main_program()
    