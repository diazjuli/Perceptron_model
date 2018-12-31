import matplotlib.pyplot as plt


def plot_error(axes, performance):
    """
    This function draws a plot for each epoch's performance

    :param axes: Matplotlib Axes
    :param performance: list of tuples: tuple shape: (epoch, # of errors, [weights], bias)
    :return: void
    """

    # get arrays containing epoch and the number of misclassified wines for each epoch
    epochs = [epoch[0] for epoch in performance]
    errors = [epoch[1] for epoch in performance]

    axes[0].set_ylabel('classification errors')
    axes[0].set_xlabel('epoch')
    axes[0].set_title('Errors as a function of epoch')
    axes[0].plot(epochs, errors)


def plot_decision_boundary(axes, performance, epoch, wine_data):
    """
    This function plots the decision boundary for our wine_data

    :param axes: Matplotlib Axes :
    :param performance: list of tuples: each tuple contains performance results for an epoch of our perceptron
    :param epoch: int
    :param wine_data: pandas Dataframe of our wine data
    :return: void
    """

    # get values for decision boundary
    w2, w1 = performance[epoch][2]
    slope = - w1 / w2
    b = performance[epoch][3]

    # get min, max from x values
    xMin = wine_data['alcohol'].min()
    xMax = wine_data['alcohol'].max()

    # Calculate y values needed to draw the line
    pHmin = slope * xMin - b / w2
    pHmax = slope * xMax - b / w2

    axes[1].set_ylim([wine_data['pH'].min(), wine_data['pH'].max()])
    axes[1].set_xlim([xMin, xMax])
    axes[1].set_title('Decision boundary on epoch: {}'.format(epoch))

    # draw line
    axes[1].plot([xMin, xMax], [pHmin, pHmax], 'b--', label='Decision boundary')
    # fill good/bad regions for given epoch
    axes[1].fill_between([xMin, xMax], [pHmin, pHmax], wine_data['pH'].min(), color='#EAFBDF')
    axes[1].fill_between([xMin, xMax], [pHmin, pHmax], wine_data['pH'].max(), color='#F8CAD0')


def plot_scatter(axes, wine_data, good_thresh, bad_thresh):
    """
    This function plots our wine data into a scatter plot
    Good quality points are drawn in green and bad quality points in red

    :param axes: matplotlib Axes
    :param wine_data: pandas dataframe
    :param good_thresh: int : observations with quality above good_thresh are labeled good
    :param bad_thresh: int : observations with quality below bad_thresh are labeled good
    :return:
    """

    # separate data into good wines(quality > good_thresh) and bad_wines (quality < bad_thresh)
    good_wine = wine_data[(wine_data['quality'] > good_thresh)]
    good_wine = good_wine.reset_index(drop=True)
    bad_wine = wine_data[(wine_data['quality'] < bad_thresh )]
    bad_wine = bad_wine.reset_index(drop=True)

    axes[1].set_ylabel('pH')
    axes[1].set_xlabel('alcohol')
    # draw scatter plots
    axes[1].scatter(good_wine.loc[:, 'alcohol'], good_wine.loc[:, 'pH'],  c='C2',
                    label='good wines (> {} score)'.format(good_thresh), marker='.')
    axes[1].scatter(bad_wine.loc[:, 'alcohol'], bad_wine.loc[:, 'pH'], c='C3',
                    label='bad wines (> {} score)'.format(bad_thresh), marker='.')
    axes[1].legend(bbox_to_anchor=(1, 1), loc=2)


def plot_performance(performance, wine_data, good_thresh, bad_thresh, epoch= -1, save_plot=False):
    """
    Plot the performance of our perceptron or adaline.
    This function will produce two plot figures:
    1) classification Errors vs Epochs
    2) Decision boundary for wine_data

    :param performance: list of tuples, each tuple contains performance information for an epoch
    :param wine_data: pandas.dataFrame
    :param good_thresh: int
    :param bad_thresh: int
    :param epoch: int that indicates the epoch that we want to display, if -1 we use the last epoch
    :param save_plot: boolean
    :return fig:
    """
    if epoch >= len(performance) or epoch < 0:
        epoch = len(performance) - 1
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    plot_error(axes, performance)
    plot_decision_boundary(axes, performance, epoch, wine_data)
    plot_scatter(axes, wine_data, good_thresh, bad_thresh)

    if save_plot:
        plt.savefig('./performance_plot.png')

    plt.show(fig)
    return fig



