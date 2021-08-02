import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from math import ceil
import pandas as pd
from matplotlib import colors as mcolors
import matplotlib.cm as cm


def get_quantiles2(df, hue_column, indexing_column, target_column, quantile=None):
    if quantile is None:
        quantile = [5, 95]

    df = df.copy()

    begin = df[indexing_column].max()
    end = df[indexing_column].min()

    space = np.linspace(begin, end)
    quantile_array = []
    for s in space:
        a = df.sort_values(by=indexing_column).groupby(hue_column).apply(
            lambda x: np.interp(s, x[indexing_column], x[target_column])).values
        q = np.percentile(a, quantile)
        quantile_array.append(q)
    quantile_array = np.array(quantile_array).reshape(-1, 2)

    return np.hstack([space.reshape(-1, 1), quantile_array])


def get_quantiles(df, hue_column, indexing_column, target_column, quantile=None):
    if quantile is None:
        quantile = [2.5, 97.5]

    df = df.copy()

    begin = df.groupby(hue_column)[indexing_column].max().min()
    end = df.groupby(hue_column)[indexing_column].min().max()

    space = np.linspace(begin, end)
    quantile_array = []
    for s in space:
        quantile_array.append(
            np.percentile(df.sort_values(by=indexing_column).groupby(hue_column)[target_column, indexing_column].apply(
                lambda x: np.interp(s, x[indexing_column], x[target_column])).values, quantile))
    quantile_array = np.array(quantile_array).reshape(-1, 2)

    return np.hstack([space.reshape(-1, 1), quantile_array])


def update_fontsize(ax, label, fontsize):
    if label == 'x':
        ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
    elif label == 'y':
        ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
    return ax


def get_quantile_per_sample(df, quantile, base_column):
    """Calculates the specified quantile across the sample."""
    df = df.copy()
    output = pd.DataFrame()
    for unique_sample in df[base_column].unique():
        object_df = df[df[base_column] == unique_sample]
        output = output.append(object_df.quantile(quantile))

    return output.sort_values(by=base_column)


def gen_color_mapping():
    return [mcolors.to_rgba((0, 1, 0)) for i in range(404)]


def plot_flights_n_boundary(df_flights, x_column, y_column, hue_column,
                            color=None, alpha=1,
                            color_column=None, color_map=None,
                            df_lower_boundary=None, df_upper_boundary=None,
                            xlims=None, ylims=None,
                            title=None, xlabel=None, ylabel=None,
                            lower_boundary_label=None,
                            upper_boundary_label=None,
                            highlight_flight=None,
                            highlight_flights=None,
                            label_highlighted_flights=False,
                            include_legend=True):
    """Plots pair of parameters for every flight within the compiled dataframe."""
    df_flights = df_flights.copy()
    fig, ax = plt.subplots()
    if xlims is None:
        x_min = round(min(df_flights[x_column]), -1)
        x_min_margin = round(0.1 * (round(min(df_flights[x_column]), -1)) -
                             ceil(max(df_flights[x_column])), -2)
        x_max = ceil(max(df_flights[x_column]))
        x_max_margin = round(0.1 * (ceil(max(df_flights[x_column])) -
                                    round(min(df_flights[x_column]), -1)), -2)
        xlims = (x_min + x_min_margin, x_max + x_max_margin)
    else:
        xlims = sorted(xlims)
        df_flights = df_flights[
            df_flights[x_column] >= xlims[0]][
            df_flights[x_column] <= xlims[1]]
    if ylims is None:
        ylims = (
            round(min(df_flights[y_column]), -1),
            ceil(max(df_flights[y_column]))
        )
    else:
        ylims = sorted(ylims)
        df_flights = df_flights[
            df_flights[y_column] >= ylims[0]][
            df_flights[y_column] <= ylims[1]
            ]
    if xlabel is None:
        xlabel = x_column
    if ylabel is None:
        ylabel = y_column
    # if title is None:
    #     title = y_column+' vs '+x_column
    if lower_boundary_label is None:
        lower_boundary_label = 'Lower boundary'
    if upper_boundary_label is None:
        upper_boundary_label = 'Upper boundary'

    if color is None:
        if color_map is None:
            color = None
        else:
            color = [color_map[i] for i in color_column]
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    array_collection = [
        np.transpose(
            np.column_stack(
                df_flights[df_flights[hue_column] == flight_id]
                [[x_column, y_column]].to_numpy())
        ) for flight_id in df_flights[hue_column].unique()
    ]
    line_segments = LineCollection(array_collection,
                                   linewidths=(0.5, 1, 1.5, 2),
                                   linestyles='solid',
                                   alpha=alpha,
                                   color=color,
                                   label='Flights')
    line_segments.set_array(np.arange(len(array_collection)))
    ax.add_collection(line_segments)

    if (df_lower_boundary is not None) & (df_upper_boundary is not None):
        _ = ax.plot(df_lower_boundary[x_column],
                    df_lower_boundary[y_column],
                    color='black',
                    linewidth=2.5,
                    label=lower_boundary_label)
        _ = ax.plot(df_upper_boundary[x_column],
                    df_upper_boundary[y_column],
                    color='black',
                    linewidth=2.5,
                    label=upper_boundary_label)
        # ax.fill_betweenx(df_lower_boundary[y_column],
        #                  df_lower_boundary[x_column],
        #                  df_upper_boundary[x_column],
        #                  color="black",
        #                  alpha=0.2)
        ax.fill_between(df_lower_boundary[x_column],
                        df_lower_boundary[y_column],
                        df_upper_boundary[y_column],
                        color="black",
                        alpha=0.2)

    if highlight_flight is not None:
        _ = ax.plot(df_flights[df_flights[hue_column] ==
                               highlight_flight][x_column],
                    df_flights[df_flights[hue_column] ==
                               highlight_flight][y_column],
                    color='red',
                    linewidth=1.25,
                    label=highlight_flight)

    if highlight_flights is not None:
        for flight in highlight_flights:
            _ = ax.plot(df_flights[df_flights[hue_column] ==
                                   flight][x_column],
                        df_flights[df_flights[hue_column] ==
                                   flight][y_column],
                        color='red',
                        linewidth=1.25,
                        label=flight if label_highlighted_flights else "_nolegend_")

    if include_legend:
        ax.legend()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    plt.show()

    return fig, ax


def plot_silhouette(sample_silhouette_values, n_clusters, labels, silhouette_avg):
    """
    Plots the silhouette of the clustering process.

    Ref:
        https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

    Parameters
    ----------
    sample_silhouette_values : numpy.ndarray
        Silhouette coefficient for each sample.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : axis
        Matplotlib axis object.

    """
    fig, ax = plt.subplots()

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks(np.linspace(-1, 1, 11))

    plt.show()

    return fig, ax


if __name__ == '__main__':
    df_flights = pd.DataFrame(
        {'id': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], 'x': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
         'y': [1, 2, 3, 4, 2, 4, 6, 8, 4, 8, 10, 12, 5, 2, 6, 8]})
    quantile_10 = get_quantile_per_sample(df_flights, 0.1, "x")
    quantile_90 = get_quantile_per_sample(df_flights, 0.9, "x")
    _ = plot_flights_n_boundary(df_flights,
                                'x', 'y', 'id',
                                df_lower_boundary=quantile_10,
                                df_upper_boundary=quantile_90,
                                lower_boundary_label='10th percentile',
                                upper_boundary_label='90th percentile'
                                )
