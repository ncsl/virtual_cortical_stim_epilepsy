import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy.stats
import sklearn.preprocessing
from mpl_toolkits.axes_grid1 import make_axes_locatable

from cortstim.edv.base.config.config import FiguresConfig
from cortstim.edv.baseplot import BasePlotter


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


class PlotHeatmap(BasePlotter):
    def __init__(self, figure_dir):
        super(PlotHeatmap, self).__init__(figure_dir=figure_dir)

    def plot_fragility_map(self, fragmat, labels, onsetchans, timepoints=None,
                           spreadchans=[], vertlines=[], vertlabels=[],
                           titlestr="Fragility Map iEEG",
                           figure_name="Fragility Map iEEG",
                           fontsize=FiguresConfig.LARGE_FONT_SIZE, markercolors=['r', 'k'],
                           figsize=FiguresConfig.LARGE_SIZE, cbarlabel="Fragility Metric",
                           save=True):
        onsetinds = [ind for ind, ch in enumerate(labels) if ch in onsetchans]
        spreadinds = [ind for ind, ch in enumerate(labels) if ch in spreadchans]
        spreadinds = list(set(spreadinds) - set(onsetinds))
        indicecolors = [onsetinds, spreadinds]
        colors = ['red', 'blue']

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax = PlotHeatmap.plot_heatmap_overtime(fragmat, ylabels=labels,
                                                        indicecolors=indicecolors, colors=colors,
                                                        # indices_red_y=onsetinds,
                                                        subplot=ax,
                                                        fontsize=fontsize,
                                                        titlestr=titlestr,
                                                        cbarlabel=cbarlabel)
        for i, line in enumerate(vertlines):
            if vertlabels:
                label = vertlabels[i]
            else:
                label = None
            self.plotvertlines(ax, line, color=markercolors[i], label=label)

        if timepoints is not None:
            xticks = np.array(ax.get_xticks()).astype(int)[1:-1]
            newxticks = timepoints[xticks, 0]
            ax.set_xticklabels(newxticks)
            ax.set_xticks(xticks)

        if save:
            print("Saving to: ", figure_name)
            fig.tight_layout()
            self.save_figure(fig, "{}".format(figure_name))

        return fig, ax

    def plot_scalpeeg_topimage(self, datavec, rawinfo, index, overall_index, onsetwin=None,
                               offsetwin=None, save=True,
                               titlestr="Scalp eeg",
                               figure_name="scalp eeg movie", cbarlabel="Fragility Metric"):
        # initialize figure, grid and axes
        fig = plt.figure(figure_name, figsize=(10, 10))
        grid = plt.GridSpec(3, 2, wspace=0.4, hspace=0.3)
        axes = plt.subplot(grid[0:2, :])

        # plot the topographic map using the data matrix
        imax, cn = mne.viz.plot_topomap(datavec, pos=rawinfo, cmap='jet',
                                        show=False, axes=axes, vmin=0, vmax=1)

        color = 'black'
        if index >= onsetwin and index < offsetwin:
            color = 'red'

        # set the title
        axes.set_title(titlestr)

        # set the colormap and its axes
        cmap = plt.set_cmap('jet')
        divider = make_axes_locatable(axes)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(imax, cax1)
        cbar.set_label(cbarlabel)

        # make progress bar
        ax2 = plt.subplot(grid[2, :])
        x = np.arange(overall_index)
        y = [1] * len(x)
        line = matplotlib.lines.Line2D(x, y, lw=5., color='r', alpha=0.4)
        ax2.add_line(line)

        # add the progress line.
        # XXX consider using axvline
        max_height = 1.25
        y1 = [0.75, max_height]
        x1 = [index, index]
        line, = ax2.plot(x1, y1, color=color, animated=True)

        # make axes
        ax2.axis('off')
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)

        fig.tight_layout()

        if save:
            # save figures and close
            self.save_figure(fig, 'img{}.png'.format(figure_name))
            plt.close()

        return fig, axes

    def create_scalpeeg_topmovie(self, datamat, rawinfo, tempfigdir,
                                 onsetwin=None, offsetwin=None,
                                 titlestr="",
                                 figure_name=""):
        for i in range(datamat.shape[1]):
            # set a dynamic title around the seizure
            if i >= onsetwin and i < offsetwin:
                titlestr = "Seizure begins now!"
            elif i == offsetwin:
                titlestr = "Seizure ends now!"
            # print(i, titlestr)

            figure_name = str(i)

            # plot the individual time window
            fig, axes = self.plot_scalpeeg_topimage(datamat[:, i], rawinfo, index=i,
                                                    overall_index=datamat.shape[1],
                                                    onsetwin=onsetwin,
                                                    offsetwin=offsetwin,
                                                    titlestr=titlestr,
                                                    figure_name=figure_name)

    def satellite_summary(self, fragmat, labels, timepoints=[], ezcontacts=[], vertlines=[],
                          titlestr="visualize Summary ",
                          figure_name="fragility_summary_satellite_plot"):
        """
        Function for plotting a satellite summary with the original fragility map, thresholded map
        and row and column metric summaries.

        :param fragmat:
        :param labels:
        :param timepoints:
        :param ezcontacts:
        :param vertlines:
        :param titlestr:
        :param figure_name:
        :return:
        """
        fig = plt.figure(figure_name, FiguresConfig.SUPER_LARGE_PORTRAIT)

        grid = plt.GridSpec(7, 7, wspace=0.4, hspace=0.3)

        # apply moving average filter to smooth out stuff
        fragmat = np.array([movingaverage(x, 20) for x in fragmat])[:, 10:-10]

        # apply natural order to fragility labels
        # fragmat, labels = self._nat_order_labels(fragmat, labels)

        # plot original heatmap
        ax = self.plot_heatmap_overtime(fragmat, ylabels=labels, subplot=grid[0:2, 0:3],
                                        titlestr=titlestr + " Original",
                                        cbarlabel="visualize Metric")
        ax.set_xlabel('Time (sec)', fontsize=FiguresConfig.LARGE_FONT_SIZE)
        ax.set_ylabel("Channels", fontsize=FiguresConfig.LARGE_FONT_SIZE)
        for ytick in ax.get_yticklabels():
            if any(c in ytick.get_text() for c in ezcontacts):
                ytick.set_color('red')
        for line in vertlines:
            self.plotvertlines(ax, line, color='k')

        # plot thresholded map
        toplot = fragmat.copy()
        toplot[toplot < 0.6] = 0
        toplot = sklearn.preprocessing.MinMaxScaler().fit_transform(toplot)
        ax = self.plot_heatmap_overtime(toplot, ylabels=labels, subplot=grid[2:4, 0:3],
                                        titlestr=titlestr + " Thresholded Map",
                                        cbarlabel="visualize Metric")
        ax.set_xlabel('Time (sec)', fontsize=FiguresConfig.LARGE_FONT_SIZE)
        ax.set_ylabel("Channels", fontsize=FiguresConfig.LARGE_FONT_SIZE)
        for ytick in ax.get_yticklabels():
            if any(c in ytick.get_text() for c in ezcontacts):
                ytick.set_color('red')
        for line in vertlines:
            self.plotvertlines(ax, line, color='k')

        # plot metrics
        xticks = np.arange(len(labels))

        def normrowsum(x, dim):
            return np.sum(
                x, axis=dim) / np.max(np.sum(x, axis=1))

        def normcvar(x, dim):
            return np.var(
                x, axis=dim) / np.max(np.var(x, axis=1))

        def normvar(x, dim):
            x = scipy.stats.variation(x, axis=dim, nan_policy='omit') / np.nanmax(
                scipy.stats.variation(x, axis=1, nan_policy='omit'))
            x[np.argwhere(np.isnan(x))] = 0
            return x

        '''
        Summarize row metrics for each matrix

        # plot these below
        '''
        to_compute_mats = [fragmat, toplot]
        to_compute_metrics = [normrowsum, normvar, normcvar]
        metric_labels = ["Rowsum", "Variance", "Coeff Variation"]
        for i, mat in enumerate(to_compute_mats):
            for j, metric_func in enumerate(to_compute_metrics):
                ax = self.plot_vector(metric_func(mat, 1), title='', labels=labels, flipy=True,
                                      subplot=grid[2 * i:(i + 1) * 2, j + 3])
                # ax.set_yticks(xticks)
                # ax.set_yticklabels(labels, fontsize=FiguresConfig.NORMAL_FONT_SIZE, rotation=90)
                ax.set_xlabel(
                    metric_labels[j], fontsize=FiguresConfig.LARGE_FONT_SIZE)

        '''
        Summarize col metrics for each matrix

        # plot these below
        '''
        # apply moving average filter to smooth out stuff
        fragmat = np.array([movingaverage(x, 20) for x in fragmat])[:, 10:-10]
        to_compute_mats = [fragmat, toplot]
        to_compute_metrics = [normrowsum, normvar, normcvar]
        metric_labels = ["Rowsum", "Variance", "Coeff Variation"]
        for i, mat in enumerate(to_compute_mats):
            for j, metric_func in enumerate(to_compute_metrics):
                ax = self.plot_vector(metric_func(mat, 0), title='', labels=labels, flipy=False,
                                      subplot=grid[4 + j, i * 3:(i + 1) * 3])
                # ax.set_xticks(xticks)
                # ax.set_xticklabels(labels, fontsize=FiguresConfig.NORMAL_FONT_SIZE, rotation=90)
                ax.set_ylabel(
                    metric_labels[j], fontsize=FiguresConfig.LARGE_FONT_SIZE)

        self.save_figure(fig, figure_name.replace(" ", "_").replace("\t", "_"))
        # self._check_show()
        return ax
