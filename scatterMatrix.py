import matplotlib.pyplot as plt


def plot_scatter_matrix(wine_data, good_threshold, bad_threshold, save_plot=False):
    """
    Plots a scatterplot matrix of wine_data.
    Samples with wine quality over good_threshold will be plotted int a different
    color than samples with wine quality below bad_threshold.
    If save_plot is True, we will save a copy of the plot as a .png file.

    @param pd.DataFrame wine_data
    @param int good_threshold
    @param int bad_threshold
    @param boolean save_plot=False

    @return matplotlib.pyplot.figure plot
    """
    num_observations, num_attributes = wine_data.shape
    fig, axes = plt.subplots(nrows=num_attributes, ncols=num_attributes, figsize=(18,18))
    fig.subplots_adjust(hspace=0, wspace=0)

    for ax in axes.flat:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    attributes = wine_data.columns
    for i in range(num_attributes):
        axes[i, i].annotate(attributes[i], (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')

    good_wine = wine_data[wine_data.quality > good_threshold]
    bad_wine = wine_data[wine_data.quality < bad_threshold]

    for i in range(num_attributes):
        for j in range(num_attributes):
            if i != j:
                axes[i, j].scatter(good_wine.iloc[:, j], good_wine.iloc[:, i], c='C3', marker='.')
                axes[i, j].scatter(bad_wine.iloc[:, j], bad_wine.iloc[:, i], c='C2', marker='.')

    if save_plot:
        plt.savefig('./plot.png')

    return fig