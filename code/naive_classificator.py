
from numpy.random import shuffle
import numpy as np

from utility import prepare_data, accuracy, categorize_and_split


class NaiveClassificator:
    def __init__(self, file, frac=0.67):
        self.train_set, self.test = prepare_data(file, frac)
        self.train = self.train_set
        self.parameters = dict()
        self.log_parameters = dict()
        self.probability = [0, 0]
        self.log_probability = [0, 0]

    def select_fraction(self, frac):
        negative = [x for x in self.train_set if x[1] == 0]
        positive = [x for x in self.train_set if x[1] == 1]
        neg_split = int(len(negative) * frac)
        pos_split = int(len(positive) * frac)
        train_set = ([x for x in negative[:neg_split]]
                     + [x for x in positive[:pos_split]])
        shuffle(train_set)
        self.train = train_set


    def estimate(self):
        self.parameters = dict()
        self.log_parameters = dict()
        self.probability = [0, 0]
        self.log_probability = [0, 0]
        features = np.array([x[0] for x in self.train])
        targets = np.array([x[1] for x in self.train])
        count = dict()
        for d in range(10):
            for c in [0, 1]:
                for j in range(9):
                    count[(c, d, j)] = 0
        y_true = np.sum(targets)
        y_false = targets.shape[0] - y_true
        for i in range(features.shape[0]):
            for j in range(features.shape[1]):
                count[(int(targets[i]),
                       int(features[i][j]) - 1, j)] += 1
        y_c = [y_false, y_true]
        for d in range(10):
            for c in [0, 1]:
                for j in range(9):
                    self.parameters[(c, d, j)] = \
                        (count[(c, d, j)] + 1) / (y_c[c] + 10)
                    self.log_parameters[(c, d, j)] = np.log(self.parameters[(c, d, j)])
        for c in [0, 1]:
            self.probability[c] = ((y_c[c] + 1) / (targets.shape[0] + 2))
            self.log_probability[c] = np.log(self.probability[c])


    def predict_one(self, X, log = True):
        prob_zero = 1
        prob_one = 1
        log_prob_zero = 0
        log_prob_one = 0
        for i in range(len(X)):
            prob_zero *= self.parameters[(0, int(X[i]) - 1, i)]
            prob_one *= self.parameters[(1, int(X[i]) - 1, i)]
            log_prob_zero += self.log_parameters[(0, int(X[i]) - 1, i)]
            log_prob_one += self.log_parameters[(1, int(X[i]) - 1, i)]

        prob_zero *= self.probability[0]
        prob_one *= self.probability[1]
        log_prob_zero += self.log_probability[0]
        log_prob_one += self.log_probability[1]



        if not log:
            return prob_one / (prob_zero + prob_one)
        return np.exp(log_prob_one - np.log(np.exp(log_prob_one) + np.exp(log_prob_zero)))

    def decide_one(self, X):
        return self.predict_one(X) >= 0.3

    def check_accuracy(self):
        def checker(x):
            return self.decide_one(x)
        features = np.array([x[0] for x in self.test])
        targets = np.array([x[1] for x in self.test])
        return accuracy(checker, features, targets)

    def compare(self):
        def predict(x):
            return self.predict_one(x)
        X = np.array([x[0] for x in self.test])
        Y = np.array([x[1] for x in self.test])
        Y_pred = np.array([predict(x) for x in X])
        Y_cmp = [(Y[i], Y_pred[i]) for i in range(len(Y_pred))]
        return Y_cmp





