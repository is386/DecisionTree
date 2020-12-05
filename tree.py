from csv import reader

import numpy as np
from scipy.stats import mode

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

CSV_NAME = "spam.data"
SEED = 0


def load_data():
    spam_data = []
    with open(CSV_NAME, newline="") as csvfile:
        csv = reader(csvfile, delimiter=" ", quotechar="|")
        for row in csv:
            spam_data.append(row[0].split(","))
    spam_data = np.asarray(spam_data).astype(np.float)
    np.random.seed(SEED)
    np.random.shuffle(spam_data)
    return spam_data


# Uses the mean to discretize continuous data
def discretize(X):
    mean = np.mean(X, axis=0)
    for i, col in enumerate(X.T):
        for j in range(len(col)):
            if X[j][i] > mean[i]:
                X[j][i] = 1
            else:
                X[j][i] = 0
    return X


# Calculates the entropy of the given labels
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    class0 = counts[0] / np.sum(counts)
    if len(values) == 2:
        class1 = counts[1] / np.sum(counts)
        return -class0 * np.log2(class0) - class1 * np.log2(class1)
    return -class0 * np.log2(class0)


def info_gain(x, y):
    # Gets the possible values of the feature and each values count
    values, counts = np.unique(x, return_counts=True)
    # Goes through each possible value of x and creates subsets where x = v
    # Calculates the entropies of these subsets
    entropies = []
    for v in range(len(values)):
        targets = y[np.where(x == values[v])]
        e = counts[v] / np.sum(counts) * entropy(targets)
        entropies.append(e)
    return 1 - np.sum(entropies)


def get_split(data, y):
    ig = []
    # Finds the information gain of all the features
    for x in data.T:
        ig.append(info_gain(x, y))
    # Returns the index of the max ig
    return np.argmax(ig)


def make_tree(X, y, features):
    # Base cases
    if (X == X[0]).all():
        return y[len(y) - 1]
    elif X.size == 0 or len(features) == 0:
        return mode(y)[0][0]
    elif np.all(y[:] == y[0]):
        return y[0]

    # Gets the index of the feature that will be split on
    node_index = get_split(X, y)
    node = features[node_index]

    # Gets the feature to split on
    best = X[:, node_index]

    # Remove the feature to split on from the data
    next_features = features[:]
    next_features.remove(node)

    # Create a new tree
    tree = {node: {}}

    # For every possible value of the feature
    for i in np.unique(best):
        # Get the values of the data where root nodes data is i
        next_X = np.delete(X[best == i], node_index, 1)
        next_y = y[best == i]
        # Return a subtree
        tree[node][i] = make_tree(next_X, next_y, next_features)
    return tree


def predict(tree, samp):
    p = 0
    # Goes through the current node's children
    for i in tree.keys():
        # Gets the value of ith node in the testing sample
        value = samp[i]
        # Gets the subtree under the current node
        # where the value is either 0 or 1
        subtree = tree[i][value]
        # If the subtree is a tree, it will continue
        # If the subtree is 0 or 1, it will return that as the prediction
        try:
            p = predict(subtree, samp)
        except:
            return subtree
    return p


# Calculates accuracy, precision, recall, f-measure
def metrics(y, y_hat):
    hits, tp, fp, fn = 0, 0, 0, 0
    for i in range(y.size):
        hits += 1 if y[i] == y_hat[i] else 0
        if y[i] == 1 and y_hat[i] == 0:  # false negative
            fn += 1
        elif y[i] == 0 and y_hat[i] == 1:  # false positive
            fp += 1
        elif y[i] == 1 and y_hat[i] == 1:  # true positive
            tp += 1
    acc = hits / y.size * 100
    pre = tp / (tp + fp) * 100
    rec = tp / (tp + fn) * 100
    f = (2 * pre * rec) / (pre + rec)
    return acc, pre, rec, f


def main():
    # Process spam data
    spam_data = load_data()
    X, y = spam_data[:, 0:-1], spam_data[:, -1]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=1/3)
    scaler = StandardScaler().fit(train_x)
    train_x, test_x = scaler.transform(train_x), scaler.transform(test_x)
    train_x, test_x = discretize(train_x), discretize(test_x)
    features = list(range(len(train_x[0])))

    # Construct tree
    tree = make_tree(train_x, train_y, features)

    # Make predictions based on tree
    y_hat = np.asarray([predict(tree, test_x[i])
                        for i in range(test_x.shape[0])])

    results = metrics(test_y, y_hat)
    print("Accuracy:", results[0])
    print("Precision:", results[1])
    print("Recall:", results[2])
    print("F-measure:", results[3])


if __name__ == "__main__":
    main()
