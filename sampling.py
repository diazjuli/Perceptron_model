def holdout_method(wine_data, ratio=0.7):
    """
    This function uses the holdout method to partition our wine_data into two sets.
    The first set is the training data, and the second set is our validation data.
    The parameter ratio specifies the proportion of training to validation data

    :param wine_data: pandas dataframe
    :param ratio:
    :return: tuple: shape: (training data, validation data)
    """

    train = wine_data.sample(frac=ratio)
    validation = wine_data.drop(train.index)
    return train, validation


def k_fold_dataset(wine_data, k, shuffle=False):
    """
    This function generates a k-fold cross-validation dataset from the wine_data pandas dataframe.

    :param wine_data: pandas dataframe: data that we want to separate into a cross-validation dataset
    :param k: int: indicates the number of folds we watnt o separate our data into
    :param shuffle: boolean: True if we want to randomize our dataset before separating it into k folds
    :return: list of tuples: tuple shape: (training_pandas_dataframe, validation_pandas_dataframe)
    """
    if shuffle:
        wine_data = wine_data.sample(frac=1)

    datasets = []
    fold_size = int(wine_data.shape[0] / k)

    for i in range(k):
        validation_set = wine_data.iloc[i * fold_size: (i + 1) * fold_size, :]
        datasets.append((wine_data.drop(validation_set.index), validation_set))

    return datasets


def validate_model(adaline, wine_data, k, num_epochs=300, learning_rate=0.001):
    """
    This function will first generate a k-fold cross-validation dataset of wine_data.
    Then we will train our dataset on each fold's training set and test our model on the validation set.
    We will save the accuracy for each fold.
    Finally we return the mean of the accuracies.

    :param adaline: Adaline model
    :param wine_data: Pandas dataframe that we want to model
    :param k: int: specifies how many folds we want to use for our k-folds cross validation
    :param num_epochs: int: the number of epochs that we want to train our data for
    :param learning_rate: specifies the learning rate for our Adaline model
    :return:
    """

    k_fold_sets = k_fold_dataset(wine_data, k, True)
    accuracy = []

    for fold in k_fold_sets:
        X = fold[0].loc[:, ['pH', 'alcohol']]
        X = X.values[:]
        labels = fold[0].loc[:, 'label']
        adaline.train(X, labels, num_epochs, learning_rate, False)
        accuracy.append(adaline.validate_weights(fold[1].loc[:, ['pH', 'alcohol']].as_matrix(), fold[1].loc[:, 'label']))
        print('model accuracy: {}'.format(accuracy[-1]))

    print('Mean model accuracy: {} - {} folds, lr: {}, num_epochs: {}'.format(sum(accuracy) / len(accuracy), k,
                                                                              learning_rate, num_epochs))
    return sum(accuracy) / len(accuracy)
