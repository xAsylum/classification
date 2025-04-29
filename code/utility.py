from collections import Counter

from numpy.random import shuffle
import numpy as np
import matplotlib.pyplot as plt

def parse_data(name):
    data = []
    with open(name, 'r') as file:
        while True:
            line = file.readline().strip()
            if not line:
                break
            parts = line.split()
            numbers = list(map(int, parts[:9]))
            category = int(parts[9])
            data.append((numbers, category))
    return data

def categorize_and_split(data, frac=0.67):
    negative = [x[0] for x in data if x[1] == 2]
    positive = [x[0] for x in data if x[1] == 4]
    shuffle(positive)
    shuffle(negative)

    neg_split = int(len(negative) * frac)
    pos_split = int(len(positive) * frac)
    train_set = [(x, 0) for x in negative[:neg_split]] + [(x, 1) for x in positive[:pos_split]]
    test_set = [(x, 0) for x in negative[neg_split:]] + [(x, 1) for x in positive[pos_split:]]

    shuffle(train_set)
    shuffle(test_set)

    return train_set, test_set

def prepare_data(name, frac):
    return categorize_and_split(parse_data(name), frac)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def how_good(precision, sensitivity):
    return (2 * precision * sensitivity) / (precision + sensitivity)

def accuracy(predict, X, Y):
    Y_pred = np.array([predict(x) for x in X])
    assert Y_pred.shape == Y.shape
    total = 0
    true_positive = 0
    false_negative = 0
    false_positive = 0
    true_negative = 0
    for i in range(Y_pred.shape[0]):
        if Y_pred[i] == 0 and Y[i] == 1:
            false_negative += 1
        elif Y_pred[i] == 1 and Y[i] == 0:
            false_positive += 1
        elif Y_pred[i] == 0 and Y[i] == 0:
            true_negative += 1
        else:
            true_positive += 1

        total += 1

    accuracy = (true_positive + true_negative ) / total
    precision = true_positive / (true_positive + false_positive)
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = (true_negative) / (true_negative + false_positive)
    return accuracy, precision, sensitivity, specificity

def plot_data(X, Y, label, xlabel, ylabel):
    plt.figure(figsize=(12, 5))
    plt.plot(X, Y, marker='o', color='tab:orange')
    plt.title(label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale('log')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.xticks(X, [x for x in X])
    plt.tight_layout()
    plt.show()


def plot_count(data, i):
    values = [row[0][i] for row in data]

    value_counts = Counter(values)

    keys = sorted(value_counts.keys())
    counts = [value_counts[k] for k in keys]

    plt.bar(keys, counts)

    bars = plt.bar(keys, counts, color='orange', edgecolor='black')
    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, str(height),
                 ha='center', va='bottom', fontsize=9)

    plt.xlabel(f'Value of {i + 1}-th feature')
    plt.ylabel('Count')
    plt.title(f'Distribution of features at index {i + 1}')
    plt.xticks(keys)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
