from collections import defaultdict
import numpy as np
from nltk.tokenize import word_tokenize


class NaiveBayes:
    def __init__(self,a):
        self.a=a

    def fit(self, train, labels):
        self.classes = np.unique(labels)
        self.train = train
        self.labels = labels
        self.dic = np.array([defaultdict(lambda: 0) for index in range(self.classes.shape[0])])
        st = open("sw.txt", "r", encoding="utf_8")

        stopwords = []
        for i in st.readlines():
            stopwords.append(i.strip())
        for index2, value1 in enumerate(self.classes):
            ca = []
            for index, value in enumerate(self.labels):
                if value == value1:
                    ca.append(self.train[index])
            clean_text = []
            for i in ca:
                word_tokens = word_tokenize(i)
                # print(word_tokens)
                filtered_sentence = [w for w in word_tokens if not w in stopwords]
                # print(filtered_sentence)
                clean_text.append(filtered_sentence)
            # print(clean_text)

            for i in clean_text:
                for token_word in i:  # for every word in preprocessed example

                    self.dic[index2][token_word] += 1

        self.p_c = np.zeros(self.classes.shape[0])
        all_words = []
        cat_word_counts = np.zeros(self.classes.shape[0])
        for index, value3 in enumerate(self.classes):
            counter = 0
            for j in self.labels:
                if j == value3:
                    counter += 1
            self.p_c[index] = counter / self.labels.shape[0]

            cat_word_counts[index] = np.sum(
                np.array(list(self.dic[index].values())))
            print(cat_word_counts[index])
            all_words += self.dic[index].keys()

        self.vocab = np.unique(np.array(all_words))
        self.vocab_length = self.vocab.shape[0]
        print(self.vocab.shape[0])
        self.d = []
        for i in range(len(self.classes)):
            self.d.append(cat_word_counts[i] + self.vocab_length + self.a)



    def predict(self, test_set):
        st = open("sw.txt", "r", encoding="utf_8")

        stopwords = []
        for i in st.readlines():
            stopwords.append(i.strip())
        predictions = []
        for i in test_set:
            word_tokens = word_tokenize(i)
            filtered_sentence = [w for w in word_tokens if not w in stopwords]
            Lprob = np.zeros(self.classes.shape[0])
            prop = np.zeros(self.classes.shape[0])
            for index, value in enumerate(self.classes):

                for test_token in filtered_sentence:

                    test_token_prob = (self.dic[index].get(test_token, 0) + self.a) / self.d[index]
                    Lprob[index] += np.log(test_token_prob)
                prop[index] = Lprob[index] + np.log(self.p_c[index])
            predictions.append(self.classes[np.argmax(prop)])

        return np.array(predictions)
