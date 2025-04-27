import numpy as np
from numpy.random import shuffle

from utility import prepare_data, sigmoid, accuracy

class LogisticRegression:
    def __init__(self, file, frac = 0.67, lr = 0.01, n = 1000):
        self.train_set, self.test = prepare_data(file, frac)
        self.train = self.train_set
        self.theta = np.zeros(10)
        self.learning_rate = lr
        self.iterations = n
        self.learning_curve = []

    def select_fraction(self, frac):
        negative = [x for x in self.train_set if x[1] == 0]
        positive = [x for x in self.train_set if x[1] == 1]
        neg_split = int(len(negative) * frac)
        pos_split = int(len(positive) * frac)
        train_set = ([x for x in negative[:neg_split]]
                     + [x for x in positive[:pos_split]])
        shuffle(train_set)
        self.train = train_set


    def hypothesis(self, x):
        return sigmoid(x @ self.theta)

    def cost(self, planning_matrix, targets, regularization_coef = 0.0):
        def cost_one(x, y):
            return -y * np.log(self.hypothesis(x)) - (1 - y) * np.log(1 - self.hypothesis(x))

        return ((
            np.mean([cost_one(planning_matrix[i], targets[i]) for i in range(len(targets))]))
                + regularization_coef * self.theta.T.dot(self.theta))
    def predict(self, X):
        return self.hypothesis(X) >= 0.5

    def fit(self, regularization_coef = 0.0):
        features = np.array([x[0] for x in self.train])
        planning_matrix = np.hstack((np.ones((features.shape[0], 1)), features))
        targets = np.array([x[1] for x in self.train])
        self.learning_curve = []
        self.theta = np.zeros(planning_matrix.shape[1])

        for _ in range(self.iterations):
            predictions = np.array([self.hypothesis(row) for row in planning_matrix])
            errors = predictions - targets
            gradient = (planning_matrix.T @ errors) / len(targets)
            self.theta -= self.learning_rate * gradient + regularization_coef * self.theta
        #print(self.cost(planning_matrix, targets, regularization_coef))

    def check_accuracy(self):
        def checker(X):
            return self.predict(X)
        features = np.array([x[0] for x in self.test])
        planning_matrix = np.hstack((np.ones((features.shape[0], 1)), features))
        targets = np.array([x[1] for x in self.test])

        return accuracy(checker, planning_matrix, targets)