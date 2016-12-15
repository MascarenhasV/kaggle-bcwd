import argparse
import heapq
import random
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold



# loads the data from the training and test files and return features and labels
def  load_data(file_name):
    # load the  data
    dataframe = pd.read_csv(file_name,header=None, skiprows=1, usecols=range(1, 32))
    feature_size = dataframe.shape[1]-1
    # choose all except last column as features
    features = dataframe.ix[0:, 2:feature_size+1]
    # choose the last column as the labels
    labels = dataframe.ix[0:, 1]

    return features, labels


# generates a mask to filter the data
def generate_mask(data, init_size):
    # randomly sample data
    indices = random.sample(range(0, data.shape[0]), init_size)
    # create mask for the indices in the data
    mask = data.index.isin(indices)

    return mask

# converts the dataframe and series to numpy arrays, and apply mask to split data
# into labeled pool and unlabeled pool
def prepare_data(data_train, labels_train, mask):

    # split the data in labeled and unlabeled data using the mask
    labeled_instances = np.array(data_train[mask])
    labeled_instances_labels = np.array(labels_train[mask])
    unlabeled_instances = np.array(data_train[~mask])
    unlabeled_instances_labels = np.array(labels_train[~mask])

    return labeled_instances, labeled_instances_labels, unlabeled_instances, unlabeled_instances_labels

# create a dictionary of all the labels:
# '0':'Actin', '1':'Endoplasmic_Reticulum', '2':'Endosomes', '3':'Lysosome','4':'Microtubules','5':'Mitochondria',
# '6':'Peroxisomes', '7':'Plasma_Membrane'
# these labels are only used for identification when we find the point's labels using the distance metric
def prepare_label_dict(labels):
    labels = sorted(list(set(labels)))
    label_dict = {}
    for index in range(0, len(labels)):
        label_dict[index] = labels[index]

    return  label_dict

# active learner implementation
def active_learner(labeled_instances, labeled_instances_labels, unlabeled_instances,
                   unlabeled_instances_labels, label_dict, batch_size, features_size):

    # create a new classifier instance
    #clf = SVC(random_state=0, kernel='linear',probability=True)

    clf = AdaBoostClassifier(
        learning_rate=1,
        n_estimators=400,
        algorithm="SAMME.R")

    # get all the indices of the features
    feature_indices = list(xrange(labeled_instances.shape[1]))
    # if we have more features in the data than the actual number of true features,
    # then perform feature selection using SelectKBest
    # fit the model using only those features which have been found to be useful
    if labeled_instances.shape[1] > features_size:
        selector = SelectKBest(f_classif, k=features_size)
        selector.fit(labeled_instances, labeled_instances_labels)
        feature_indices = selector.get_support(indices=True)
        new_labeled_instances = labeled_instances[:, feature_indices]
        clf.fit(new_labeled_instances, labeled_instances_labels)
    else:
        clf.fit(labeled_instances, labeled_instances_labels)

    # find the distances for the data from each hyperplane
    labels_dist = clf.predict_proba(unlabeled_instances[:, feature_indices])

    # initialize the arrays to store indices which would be queried
    query_indices = []
    query_indices_labels = []
    data_heap = []

    # iterate on the distances for all points, find the difference of distances of the top two points
    # in the prediction and push them to a heap
    for index in range(0, len(labels_dist)):
        current_row = np.array(labels_dist[index])
        inferred_label = label_dict[np.argmax(current_row)]
        top_two = np.sort(current_row)[::-1][0:2]
        difference = top_two[0] - top_two[1]
        heapq.heappush(data_heap, (np.abs(difference), index, inferred_label))

    # fetch the minimum distance values from the heap
    min_gap = heapq.nsmallest(batch_size, data_heap)

    # fetch the indices of the points which have been selected to be queried
    for row in min_gap:
        query_indices.append(row[1])

    # fetch the queried data and it's labels
    queried_data = unlabeled_instances[query_indices]
    queried_data_labels = unlabeled_instances_labels[query_indices]  # queried labels

    # add the queried labels and instances to the labeled instances pool
    labeled_instances = np.vstack((labeled_instances, queried_data))
    labeled_instances_labels = np.concatenate((labeled_instances_labels, queried_data_labels))

    # remove the queried labels and instances from the unlabeled instances pool
    unlabeled_instances = np.delete(unlabeled_instances, query_indices, 0)
    unlabeled_instances_labels = np.delete(unlabeled_instances_labels, query_indices, None)

    # return updated values and the classifier object
    updated_instances = labeled_instances, labeled_instances_labels, unlabeled_instances, unlabeled_instances_labels
    return clf, feature_indices, updated_instances

# random learner implementation
def random_learner(r_labeled_instances, r_labeled_instances_labels, r_unlabeled_instances,
                   r_unlabeled_instances_labels, batch_size, features_size):

    # create a new classifier instance
    #r_clf = SVC(random_state=0, kernel='linear',probability=True)

    r_clf = AdaBoostClassifier(
        learning_rate=1,
        n_estimators=400,
        algorithm="SAMME.R")

    # get all the indices of the features
    feature_indices = list(xrange(r_labeled_instances.shape[1]))
    # if we have more features in the data than the actual number of true features,
    # then perform feature selection using SelectKBest
    # fit the model using only those features which have been found to be useful
    if r_labeled_instances.shape[1] > features_size:
        selector = SelectKBest(f_classif, k=features_size)
        selector.fit(r_labeled_instances, r_labeled_instances_labels)
        feature_indices = selector.get_support(indices=True)
        new_labeled_instances = r_labeled_instances[:, feature_indices]
        r_clf.fit(new_labeled_instances, r_labeled_instances_labels)

    else:
        r_clf.fit(r_labeled_instances, r_labeled_instances_labels)

    # find the distances for the data from each hyperplane
    r_labels_predicted = r_clf.predict(r_unlabeled_instances[:, feature_indices])

    # find random indices, data and labels from the set of unlabeled points for the random learner
    random_indices = random.sample(range(0, r_unlabeled_instances.shape[0]), batch_size)
    random_instances = r_unlabeled_instances[random_indices]
    random_instance_labels = r_unlabeled_instances_labels[random_indices]

    # add the queried labels and instances to the labeled instances pool
    r_labeled_instances = np.vstack((r_labeled_instances, random_instances))
    r_labeled_instances_labels = np.concatenate((r_labeled_instances_labels, random_instance_labels))

    # remove the queried labels and instances from the unlabeled instances pool
    r_unlabeled_instances = np.delete(r_unlabeled_instances, random_indices, 0)
    r_unlabeled_instances_labels = np.delete(r_unlabeled_instances_labels, random_indices, None)

    # return updated values and the classifier object
    updated_instances = r_labeled_instances, r_labeled_instances_labels, r_unlabeled_instances, r_unlabeled_instances_labels
    return r_clf, feature_indices, updated_instances

# finds the test error for the active and random learner
def calculate_error(labels_expected, labels_predicted, r_labels_predicted):

    # error = 1 - accuracy
    active_test_error = 1.0 - metrics.accuracy_score(labels_expected, labels_predicted)
    random_test_error = 1.0 - metrics.accuracy_score(labels_expected, r_labels_predicted)

    return active_test_error, random_test_error

# print the classification report and the confusion matrices on the basis of true and predicted labels
def print_metrics(clf, labels_expected, labels_predicted):

    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(labels_expected, labels_predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels_expected, labels_predicted))

# plot figures
def plot_figures(a_test_error, r_test_error, batch_array, min_cost, min_test_error, output_fig):

    fig = plt.figure()

    plt.plot(batch_array, a_test_error, 'b-', label='Active Learner')
    plt.plot(batch_array, r_test_error, 'r-', label='Random Learner')

    # mark the lowest test error point
    axes = plt.gca()
    plt.hlines(y=min_test_error, xmin=0, xmax=min_cost, linestyles='dotted')
    plt.vlines(x=min_cost, ymin=axes.get_ylim()[0], ymax=min_test_error, linestyles='dotted')
    plt.text(min_cost, (axes.get_ylim()[0]+min_test_error)/2.0, '('+str(min_cost)+','+str(min_test_error)+')')

    fig.suptitle('Calls to oracle vs Test error', fontsize=20)
    plt.xlabel('Number of calls to oracle', fontsize=16)
    plt.ylabel('Test error', fontsize=16)

    plt.legend(loc=1)
    plt.savefig(output_fig)

# main function
def main(params):
    # fetch inputs
    data = params['data']
    batch_size = params['batch_size']
    labeled_pool_size = params['init_pool_size']
    cost = labeled_pool_size
    calls_to_oracle = params['calls_to_oracle']
    features_size = params['features_size']
    output_fig = params['output_fig']


    # load the training and the test data to get features and labels
    all_data, all_labels = load_data(data)

    data_train, data_test, labels_train, labels_expected = \
        train_test_split(all_data, all_labels, test_size=0.33, random_state=10)
    data_test = np.array(data_test)

    # generate a mask from the training data to create labeled pool instances for both the learners
    mask = generate_mask(data_train, labeled_pool_size)

    # initialize labeled and unlabeled pool instances for active learner
    labeled_instances, labeled_instances_labels, unlabeled_instances, unlabeled_instances_labels = \
        prepare_data(data_train, labels_train, mask)

    # initialize labeled and unlabeled pool instances for random learner
    r_labeled_instances, r_labeled_instances_labels, r_unlabeled_instances, r_unlabeled_instances_labels = \
        prepare_data(data_train, labels_train, mask)

    # dictionary of all labels
    label_dict = prepare_label_dict(labels_train)

    # condense all data to a variable so be fed to the functions calling the active and random learners
    instances = labeled_instances, labeled_instances_labels, unlabeled_instances, unlabeled_instances_labels
    r_instances = r_labeled_instances, r_labeled_instances_labels, r_unlabeled_instances, r_unlabeled_instances_labels

    # initialize arrays for holding data to be plotted
    a_test_error = []
    r_test_error = []
    batch_array = []

    # initialize values to be used to find the best active learner model
    min_test_error = 1
    min_cost = 0
    best_model = None

    # iterate till we run out of money
    while cost < calls_to_oracle and len(unlabeled_instances) > 0:

        if unlabeled_instances.shape[0] < batch_size:
            batch_size = unlabeled_instances.shape[0]

        # call the active learner
        clf, feature_indices, instances = active_learner(labeled_instances, labeled_instances_labels, unlabeled_instances,
                       unlabeled_instances_labels, label_dict, batch_size, features_size)

        # update cost
        cost = cost + batch_size

        # call the random learner
        r_clf, r_feature_indices, r_instances = random_learner(r_labeled_instances, r_labeled_instances_labels, r_unlabeled_instances,
                       r_unlabeled_instances_labels, batch_size, features_size)

        # predict the data on the test data set
        labels_predicted = clf.predict(data_test[:,feature_indices])
        r_labels_predicted = r_clf.predict(data_test[:,r_feature_indices])

        # calculate the errors for the active and random learner
        active_test_error, random_test_error = calculate_error(labels_expected, labels_predicted, r_labels_predicted)

        # append error values to be plotted later
        a_test_error.append(active_test_error)
        r_test_error.append(random_test_error)

        # find the best model and minimum cost
        if active_test_error < min_test_error:
            min_cost = cost
            min_test_error = active_test_error
            best_model = deepcopy(clf)

        # append cost value to be plotted later
        batch_array.append(cost)

        print "Cost: "+str(cost)

        # update the labeled and unlabeled instances for active learner
        labeled_instances, labeled_instances_labels, unlabeled_instances, unlabeled_instances_labels = \
            instances[0], instances[1], instances[2], instances[3]

        # update the labeled and unlabeled instances for random learner
        r_labeled_instances, r_labeled_instances_labels, r_unlabeled_instances, r_unlabeled_instances_labels = \
            r_instances[0], r_instances[1], r_instances[2], r_instances[3]


    # print the metrics for the learners
    print "Predictions for the active learner :"
    print_metrics(clf, labels_expected, labels_predicted)
    print "Predictions for the random learner :"
    print_metrics(r_clf, labels_expected, r_labels_predicted)

    # plot the test accuracies and save it to a file
    plot_figures(a_test_error, r_test_error, batch_array, min_cost, min_test_error, output_fig)

if __name__ =="__main__":
    np.random.seed(0)
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data', type=str, default='../data/data.csv')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=30)
    parser.add_argument('--init_pool_size', dest='init_pool_size', type=int, default=30)
    parser.add_argument('--calls_to_oracle', dest='calls_to_oracle', type=int, default=400)
    parser.add_argument('--features_size', dest='features_size', type=int, default=30)
    parser.add_argument('--output_fig', dest='output_fig', type=str, default='plot.png')
    params = vars(parser.parse_args())
    main(params)