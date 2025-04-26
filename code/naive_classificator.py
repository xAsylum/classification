from utility import prepare_data

class NaiveClassificator:
    def __init__(self, file, frac=0.67):
        self.train_set, self.test = prepare_data(file, frac)
        self.train = self.train_set

    def select_fraction(self, frac):
        length = int(len(self.train_set) * frac)
        self.train = self.train_set[:length]

    def estimate(self):
        pass

    def predict(self, X):
        pass


