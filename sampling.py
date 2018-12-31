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
