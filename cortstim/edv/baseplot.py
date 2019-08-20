import os
import warnings

import matplotlib as mp
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from natsort import index_natsorted, order_by_index

try:
    import brewer2mpl
except ImportError as e:
    print(e)

from cortstim.base.utils.data_structures_utils import ensure_list
from cortstim.base.utils.data_structures_utils import generate_region_labels
from cortstim.edv.base.config.config import FiguresConfig


# outcome_dabest.mean_diff.plot()
# dabest.TwoGroupsEffectSize

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))

    e.g. plt.imshow(ras, cmap=cmap, clim=(elev_min, elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min, vmax=elev_max))
        plt.colorbar()
        plt.show()
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


class AbstractBasePlot():
    """
    Abstract base plotting class that houses all logistical settings for a figure.

    Does not provide any functionality in terms of plotting. Helps with formatting figure,
    setting font sizes, showing it, saving it.
    """

    @staticmethod
    def figure_filename(fig=None, figure_name=None):
        if fig is None:
            fig = plt.gcf()
        if figure_name is None:
            figure_name = fig.get_label()
        # replace all unnecessary characters
        figure_name = figure_name.replace(
            ": ", "_").replace(
            " ", "_").replace(
            "\t", "_").replace(
            ",", "")
        return figure_name

    def save_vid(self, figdir, outdir, figname, prefix='img'):
        print("Command: ffmpeg -r 1 -i {figdir}/{prefix}%01d.png -vcodec mpeg4 -y {outdir}/{figname}".format(
            prefix=prefix,
            figdir=figdir,
            outdir=outdir,
            figname=figname))
        os.system("ffmpeg -r 1 -i {figdir}/{prefix}%01d.png -vcodec mpeg4 -y {outdir}/{figname}".format(prefix=prefix,
                                                                                                        figdir=figdir,
                                                                                                        outdir=outdir,
                                                                                                        figname=figname))

    def _check_show(self):
        if FiguresConfig.SHOW_FLAG:
            mp.use('TkAgg')
            plt.ion()
            plt.show()
        else:
            mp.use('Agg')
            plt.ioff()
            plt.close()

    def format_figure(self, fig, axes, fontsize):
        axes = ensure_list(axes)

        for ax in axes:
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize / 2.0)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize / 2.0)

    def _nat_order_labels(self, mat, labels):
        # get the natural order indices
        natindices = index_natsorted(labels)

        # order the matrix
        ordered_mat = np.array(order_by_index(mat, natindices))
        ordered_labels = np.array(order_by_index(labels, natindices))
        return ordered_mat, ordered_labels

    def save_figure(self, fig, figure_name):
        if FiguresConfig.SAVE_FLAG:
            if "." in figure_name:
                figure_name, ext = figure_name.split('.')
            else:
                ext = self.figure_format
            # get figure name and set it with the set format
            figure_name = self.figure_filename(fig, figure_name)
            figure_name = figure_name[:np.min(
                [100, len(figure_name)])]
            # + '.' + self.figure_format
            if not (os.path.isdir(self.figure_dir)):
                os.mkdir(self.figure_dir)

            outputfilepath = os.path.join(self.figure_dir, figure_name + f".{ext}")
            plt.savefig(outputfilepath,
                        box_inches='tight')

    def set_colorbar(self, img, axes, cbarlabel):
        # set the colormap and its axes
        # cbar = plt.colorbar(img)
        # cax1 = cbar.ax

        # divider = make_axes_locatable(axes)
        # # cax1 = divider.append_axes("right", size="5%", pad=0.05)
        # cax1=None
        # cbar = plt.colorbar(img)

        # make a color bar
        divider = make_axes_locatable(axes)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(img, cax=cax1)
        return cbar, cax1


class BasePlotter(AbstractBasePlot):
    def __init__(self, figure_dir):
        self.figure_dir = figure_dir  # self.config.out.FOLDER_FIGURES
        self.figure_format = FiguresConfig.FIG_FORMAT

        # set highlighting cursor
        self.HighlightingDataCursor = lambda *args, **kwargs: None

        # get the mp backend
        if mp.get_backend() in mp.rcsetup.interactive_bk:
            # and self.config.figures.MOUSE_HOOVER:
            try:
                from mpldatacursor import HighlightingDataCursor
                self.HighlightingDataCursor = HighlightingDataCursor
            except ImportError:
                # self.config.figures.MOUSE_HOOVER = False
                warnings.warn(
                    "Importing mpldatacursor failed! No highlighting functionality in plots!")
        else:
            warnings.warn(
                "Noninteractive matplotlib backend! No highlighting functionality in plots!")
            # self.config.figures.MOUSE_HOOVER = False

    def setup_figure(self, nrow=1, ncol=1, figure_size=FiguresConfig.VERY_LARGE_SIZE):
        fig, axs = plt.subplots(nrows=nrow, ncols=ncol,
                                # sharey="row",
                                # sharex="col",
                                figsize=figure_size)
        axs = np.array(axs)
        axs = axs.reshape(-1)
        return fig, axs

    def plotvertlines(self, ax, time, color='k', label=None):
        """
        Function to plot vertical dashed lines on an axis.

        :param ax: (Axes) object to plot on
        :param time: time to plot a vertical line on
        :param color: color to make the vertical line
        :return: (Axes) object
        """
        if isinstance(time, list):
            t = time.pop()
            ax = self.plotvertlines(ax, t, color=color)

        # plot vertical lines of 'predicted' onset/offset
        ax.axvline(time,
                   color=color,
                   linestyle='dashed',
                   linewidth=10, label=label)
        return ax

    @staticmethod
    def plot_heatmap_overtime(mat, subplot, titlestr,
                              ylabels=[], xlabels=[],
                              ax=None,
                              show_y_labels=True, show_x_labels=False,
                              indicecolors=[], colors=[],
                              sharey=None,
                              fontsize=FiguresConfig.LARGE_FONT_SIZE,
                              cbarlabel="",
                              cmapname='inferno'):
        """
        Static method of base plotter for plotting a 2D heatmap.

        :param mat:
        :param subplot:
        :param titlestr:
        :param ylabels:
        :param xlabels:
        :param ax:
        :param show_y_labels:
        :param show_x_labels:
        :param indicecolors:
        :param colors:
        :param sharey:
        :param fontsize:
        :param cbarlabel:
        :param cmapname:
        :return:
        """
        assert len(indicecolors) == len(colors)

        if ax is None:
            ax = plt.subplot(subplot, sharey=sharey)  # initialize ax
        # set title
        ax.set_title(titlestr, fontsize=fontsize)

        # get the size of the matrix to plot
        mat_size = mat.shape[0]
        time_size = mat.shape[1]

        # set the yticks & color
        y_ticks = np.arange(mat_size).astype(int)

        # plot the heatmap
        # cmap = plt.set_cmap(cmapname)
        if cmapname == 'OrRd':
            bmap = brewer2mpl.get_map("OrRd", 'Sequential', 9, reverse=False)
            cmap = bmap.mpl_colormap
        elif cmapname == 'inferno':
            cmap = 'inferno'
        else:
            cmap = cmapname

        # cmap = 'viridis'
        img = ax.imshow(mat,
                        origin='lower',
                        cmap=cmap,
                        aspect='auto',
                        interpolation='nearest',
                        alpha=0.3,
                        )
        # set a grid on the plot
        ax.grid(True, color='grey')

        # set x ticks and ylabels
        if show_x_labels:
            # set the xticks & color
            x_ticks = np.array(
                np.arange(0, time_size, time_size / 10), dtype=np.int32)
            x_color = 'k'

            ax.set_xticks(x_ticks)
            ax.set_xticklabels(xlabels[x_ticks])

        # set y ticks and ylabels
        if show_y_labels:
            # get the ylabbels
            region_labels = np.array(
                ["%d. %s" % l for l in zip(range(mat_size), ylabels)])
            # region_labels = np.array(ylabels)

            ax.set_yticks(y_ticks)
            ax.set_yticklabels(region_labels, fontsize=fontsize / 1.5)

            # # check if there was only one color set
            ticklabels = ax.get_yticklabels(minor=False)

            # set colors based on lists passed in
            for inds, color in zip(indicecolors, colors):
                for idx in inds:
                    ticklabels[idx].set_color(color)
            ax.set_yticklabels(ticklabels)
        else:
            ax.set_yticklabels([])

        # set tick ylabels and markers along the heatmap x/y axis
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize / 1.5)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize / 1.5)

        # format the object correctly
        ax.autoscale(tight=True)
        # make a color bar
        cbar, cax1 = BasePlotter.set_colorbar(BasePlotter, img, ax, cbarlabel)
        cbar.set_label(cbarlabel, rotation=270,
                       fontsize=fontsize, labelpad=60)
        cax1.tick_params(labelsize=fontsize)
        return ax

    def _plot_ts(self, data, labels, ax,
                 show_ylabels=True,
                 offset=0.0,
                 special_idx=[],
                 errors_list=[],
                 fontsize=FiguresConfig.LARGE_FONT_SIZE):
        """
        Method for plotting a set of time series data.

        :param data: (np.ndarray) dataset of time series to be plotting. (Samples X Time)
        :param labels: (list) of labels (Samples x 1)
        :param ax: (Axes) object to plot on
        :param show_ylabels:
        :param offset:
        :param special_idx:
        :param fontsize:
        :return:
        """
        if data.ndim == 1:
            data = data[np.newaxis, :]
        offset = int(offset)
        # apply offset setting onto the data
        data = data[:, offset:]

        # get shape of data to be plotted
        nsamples, ntimes = data.shape

        nTS = 1
        def_alpha = 1.0
        # generate ylabels for the plot
        labels = generate_region_labels(nsamples, labels)

        # set plotting parameters: alpha_ratio, colors, alphas
        alpha_ratio = 1.0 / nsamples
        colors = np.array(['k'] * nTS)
        alphas = np.maximum(np.array(
            [def_alpha] *
            nTS) *
                            alpha_ratio,
                            1.0)
        colors[special_idx] = 'r'
        alphas[special_idx] = np.maximum(alpha_ratio, 0.1)

        # apply normalization for each trace
        for i in range(nsamples):
            data[i, :] = data[i, :] / np.nanmax(data[i, :])

            # plot each trace
            x = np.arange(ntimes)
            for itrace in range(nTS):
                for i in range(nsamples):
                    y = data[i, :] + np.r_[i]
                    ax.plot(x, y,
                            color=colors[itrace],
                            label=labels[itrace],
                            alpha=alphas[itrace])

                    # plot errors bars
                    if errors_list:
                        error = errors_list[error]
                        ax.fill_between(x, y - error, y + error,
                                        color=colors[itrace],
                                        alphas=alphas[itrace])

        if show_ylabels:
            # print("Labels are : ", labels)
            y_ticks = np.arange(len(labels))
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(labels, fontsize=fontsize / 1.5)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize / 1.5)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize / 1.5)

        return ax

    @staticmethod
    def plot_vector(vector, subplot, title, labels,
                    flipy=False,
                    sharey=None,
                    fontsize=FiguresConfig.VERY_LARGE_FONT_SIZE):
        """
        Method for plotting a vector (can be time series) into a subplot.

        :param vector:
        :param subplot:
        :param title:
        :param labels:
        :param flipy:
        :param sharey:
        :param fontsize:
        :return:
        """
        ax = plt.subplot(subplot, sharey=sharey)
        plt.title(title, fontsize=fontsize)

        n_vector = labels.shape[0]
        y_ticks = np.array(range(0, n_vector * 3, 3), dtype=np.int32)

        color = 'black'

        # plot vector
        if flipy:
            ax.plot(vector, np.arange(0, len(vector)),
                    color=color, linestyle='-')
            ax.invert_yaxis()
        else:
            ax.plot(vector, color=color, linestyle='-')

        # ax.grid(True, color='grey')

        # format the axes
        ax.autoscale(tight=True)
        return ax
