import nltk
from nltk import FreqDist
from nltk.tokenize import wordpunct_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, porter
import numpy as np
import math
import random
import pandas
from nltk.stem import *
from nltk.stem.porter import *

stemmer = PorterStemmer()


class MultinomialNaiveBayes:
    def __init__(self, nb_classes, nb_words, pseudocount):
        self.nb_classes = nb_classes
        self.nb_words = nb_words
        self.pseudocount = pseudocount

    def fit(self, X, Y):
        # Uzima broj tweet-ova
        nb_examples = X.shape[0]

        # Racunamo P(Klasa) - priors
        # np.bincount nam za datu listu vraca broj pojavljivanja svakog celog
        # broja u intervalu [0, maksimalni broj u listi]
        self.priors = np.bincount(Y) / nb_examples
        print('Priors:')
        print(self.priors)

        # Racunamo broj pojavljivanja svake reci u svakoj klasi
        occs = np.zeros((self.nb_classes, self.nb_words))
        for i in range(nb_examples):
            c = Y[i]
            for w in range(self.nb_words):
                cnt = X[i][w]
                occs[c][w] += cnt
        print('Occurences:')
        print(occs)

        # Racunamo P(Rec_i|Klasa) - likelihoods
        self.like = np.zeros((self.nb_classes, self.nb_words))
        for c in range(self.nb_classes):
            for w in range(self.nb_words):
                up = occs[c][w] + self.pseudocount
                down = np.sum(occs[c]) + self.nb_words * self.pseudocount
                self.like[c][w] = up / down
        print('Likelihoods:')
        print(self.like)

    def predict(self, bow):
        # Racunamo P(Klasa|bow) za svaku klasu
        probs = np.zeros(self.nb_classes)
        for c in range(self.nb_classes):
            prob = np.log(self.priors[c])
            for w in range(self.nb_words):
                cnt = bow[w]
                prob += cnt * np.log(self.like[c][w])
            probs[c] = prob
        # Trazimo klasu sa najvecom verovatnocom
        # print('\"Probabilites\" for a test BoW (with log):')
        # print(probs)
        prediction = np.argmax(probs)
        return prediction

    def predict_multiply(self, bow):
        # Racunamo P(Klasa|bow) za svaku klasu
        # Mnozimo i stepenujemo kako bismo uporedili rezultate sa slajdovima
        probs = np.zeros(self.nb_classes)
        for c in range(self.nb_classes):
            prob = self.priors[c]
            for w in range(self.nb_words):
                cnt = bow[w]
                prob *= self.like[c][w] ** cnt
            probs[c] = prob
        # Trazimo klasu sa najvecom verovatnocom
        print('\"Probabilities\" for a test BoW (without log):')
        print(probs)
        prediction = np.argmax(probs)
        return prediction


def clean(text):
    # Define regex pattern for links
    link_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Define regex pattern for Twitter username tags
    username_regex = r'@[\w]+'

    # Define regex pattern for words containing numbers
    word_with_number_regex = r'\b\w*\d\w*\b'
    remove_non_alpha_regex = r'[^a-zA-Z\s]+'

    # Remove URLs, usernames, numbers, and consecutive punctuation marks with empty string
    text = re.sub(link_regex, "", text)
    text = re.sub(username_regex, "", text)
    text = re.sub(word_with_number_regex, "", text)
    text = re.sub(remove_non_alpha_regex, "", text)

    return text


def occurencies(word, dict, val):
    # print(val)
    if dict.get(word) is None:
        dict[word] = [0, 0]

    if val == 1:
        dict[word][1] += 1
    else:
        dict[word][0] += 1


def freq_score(word, doc):
    return doc.count(word) / len(doc)


if __name__ == "__main__":
    file_name = 'disaster-tweets.csv'

    csvFile = pandas.read_csv(file_name, usecols=['target', 'text']).dropna()
    all_data = [[row[0], row[1]] for i, row in csvFile.iterrows()]
    random.shuffle(all_data)
    Y = [row[1] for row in all_data]
    # print(Y)
    clean_corpus = []
    nltk.download('stopwords')
    stop_punc = set(stopwords.words('english')).union(set(punctuation))

    # dictionary [neg, poz]

    for doc in all_data:
        doc[0] = clean(doc[0])
        # tokenizovao
        words = wordpunct_tokenize(doc[0])
        words_lower = [w.lower() for w in words]
        # print("lower: ", words_lower)
        words_filtered = [w for w in words_lower if w not in stop_punc]
        # print("filter: ", words_filtered)
        words_stemmed = [stemmer.stem(w) for w in words_filtered]
        # print('Final:', words_stemmed)
        clean_corpus.append(words_stemmed)
        # Primetiti razliku u vokabularu kada se koraci izostave

    vocab_set = set()
    for doc in clean_corpus:
        for word in doc:
            vocab_set.add(word)

    vocab = list(vocab_set)
    vocab = [string for string in vocab if len(string) >= 3]

    # b)
    lr_data = {}

    for doc_idx in range(len(clean_corpus)):
        doc = clean_corpus[doc_idx]
        for word_idx in range(len(doc)):
            occurencies(doc[word_idx], lr_data, Y[doc_idx])

    filtered_lr_data = {word: counts for word, counts in lr_data.items() if all(count >= 10 for count in counts)}
    result_dict = {word: [counts[0], counts[1], counts[1] / counts[0]] for word, counts in filtered_lr_data.items()}

    sorted_by_count1 = sorted(result_dict.items(), key=lambda x: x[1][1], reverse=True)
    sorted_by_count0 = sorted(result_dict.items(), key=lambda x: x[1][0], reverse=True)

    # 10 reci, 5 pozitvnih, 5 negativnih (sa najvisim brojem pojavljivanja)
    top_5_count1 = sorted_by_count1[:5]
    top_5_count0 = sorted_by_count0[:5]

    print(top_5_count0, top_5_count1)

    sorted_dict = sorted(result_dict.items(), key=lambda x: x[1][2], reverse=True)

    # 10 reci, 5 najvecih LR i 5 sa najmanjim
    top_5 = sorted_dict[:5]
    bottom_5 = sorted_dict[-5:]

    print(top_5, bottom_5)
    # LR za neku rec -> sto je LR veci ta rec je relavatnija za pozitivne tvitove, sto je manja relevatnija je za
    # negativne tvitove. Razlika izmedju ovih 10 za LR u odnosu sa onih 10 za broj pojavljivanja je u tome sto
    # nam LR govori koliko su pojedine reci bitne za pozitivni ishod u odnosu na negativni. Npr. nama se neka rec
    # moze pojavljivati 400 puta i u pozitivnom i u negativnom kontekstu, samim tim moze biti neutralna rec jer se
    # cesto koristi u oba konteksta, ali ako nam se neka rec pojavljuje 10 puta u pozitivnom i 1 put u negativnom
    # kontekstu, ona je verovatno relevatnija za pozitivni ishod, jer je vernija pozitivnom kontekstu.

    # b) end

    # Za svaki tweet pravimo vektor, koji ce imati 0, 1 u zavinosti da li ima ili nema tu rec (ceo vokabular)
    X = np.zeros((len(clean_corpus), len(vocab)), dtype=np.float32)

    for doc_idx in range(len(clean_corpus)):
        doc = clean_corpus[doc_idx]

        for word_idx in range(len(vocab)):
            word = vocab[word_idx]
            cnt = freq_score(word, doc)
            X[doc_idx][word_idx] = cnt

    # X -> feature vektor, za svaki tweet sve reci iz recnika freq
    # vocab -> recnik, cleaned words
    # clean_corpus -> clean lista tweet-ova

    class_names = ['Negativno', 'Pozitivno']

    training_list_x = X[:(int(0.8 * X.shape[0]))]
    # rezultati
    training_list_y = Y[:(int(0.8 * len(Y)))]

    test_list_x = X[(int(0.8 * X.shape[0])):]
    test_list_y = Y[(int(0.8 * len(Y))):]

    # print(len(training_list_x), len(training_list_y))
    # print(len(test_list_x), len(test_list_y))

    # nb_classes -> neg, poz  nb_word -> broj_reci
    model = MultinomialNaiveBayes(nb_classes=2, nb_words=len(vocab), pseudocount=1)
    model.fit(training_list_x, training_list_y)

    succ = 0
    i = 0
    for pom in test_list_x:
        res = model.predict(np.asarray(pom))
        if res == test_list_y[i]:
            succ += 1

        i += 1

    print("Result: ", succ/len(test_list_x))
