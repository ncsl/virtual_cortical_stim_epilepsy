import matplotlib.pyplot as plt
import numpy as np

from cortstim.edv.base.config.config import FiguresConfig
from cortstim.edv.baseplot import BasePlotter


class PlotRawTS(BasePlotter):
    def __init__(self, figure_dir):
        super(PlotRawTS, self).__init__(figure_dir=figure_dir)

    def timeseries_plot(self, data_list, labels_list,
                        errors_list=[],
                        offset=0.0,
                        special_idx=[],
                        titlestr_list=[],
                        vertlines=[],
                        figure_size=FiguresConfig.SMALL_SIZE,
                        fontsize=FiguresConfig.NORMAL_FONT_SIZE,
                        figure_name="raw_ts_traces"):
        if len(data_list) == 1:
            ncol = nrow = 1
        else:
            ncol = 2
            nrow = len(data_list) // 2

        fig, axs = plt.subplots(nrows=nrow, ncols=ncol,
                                sharey="row",
                                sharex="col",
                                # num=figure_name,
                                figsize=figure_size)
        axs = np.array(axs)
        axs = axs.reshape(-1)

        for idx in range(len(data_list)):
            data = data_list[idx]
            labels = labels_list[idx]
            ax = axs[idx]

            ax = self._plot_ts(data, labels, ax,
                               offset=offset,
                               errors_list=errors_list,
                               special_idx=special_idx,
                               fontsize=FiguresConfig.LARGE_FONT_SIZE)

            if titlestr_list:
                titlestr = titlestr_list.pop()
                ax.set_title(titlestr, fontsize=fontsize)

            for vertline in vertlines:
                self.plotvertlines(ax, vertline, color='r')

        # format final figure
        self.format_figure(fig, axs, fontsize=fontsize)

        # run saving of the figure
        self.save_figure(fig, figure_name=figure_name)
        self._check_show()

        return fig, axs

    def butterfly_plot(self, data_list,
                       labels=[], xlabels=[],
                       offset=0.0,
                       titlestr="Butterfly Plot",
                       figure_size=FiguresConfig.SMALL_SIZE,
                       fontsize=FiguresConfig.NORMAL_FONT_SIZE,
                       figure_name="butterfly_plot"):
        ncol = nrow = 1
        fig, ax = plt.subplots(nrows=nrow, ncols=ncol,
                               figsize=figure_size)

        for data in data_list:
            ax = self._plot_ts(data, labels, ax,
                               offset=offset,
                               special_idx=[], show_ylabels=False,
                               fontsize=FiguresConfig.LARGE_FONT_SIZE)

        ax.set_title(titlestr, fontsize=fontsize)
        ax.set_xlabel("Time Forward")
        ax.set_ylabel("Magnitude of Response")

        # set xlabels
        if xlabels:
            # set the xticks & color
            lendata = len(data_list[0])
            x_ticks = np.arange(0, lendata, lendata // 10).astype(np.int32)

            ax.set_xticks(x_ticks)
            ax.set_xticklabels(xlabels[x_ticks])
            ax.set_xticklabels(xlabels)

        # format final figure
        self.format_figure(fig, ax, fontsize=fontsize)

        # run saving of the figure
        self.save_figure(fig, figure_name=figure_name)
        self._check_show()

        return fig, ax

    def plot_boxplot(self, data_list,
                     labels=[],
                     titlestr="Box Plot",
                     figure_size=FiguresConfig.SMALL_SIZE,
                     fontsize=FiguresConfig.NORMAL_FONT_SIZE,
                     figure_name="box_plot"):
        ncol = nrow = 1
        fig, ax = plt.subplots(nrows=nrow, ncols=ncol,
                               figsize=figure_size)

        ax.boxplot(data_list,
                   labels=labels)

        ax.set_title(titlestr, fontsize=fontsize)
        ax.set_xlabel("Time Forward")
        ax.set_ylabel("Magnitude of Response")

        # format final figure
        self.format_figure(fig, ax, fontsize=fontsize)

        # run saving of the figure
        self.save_figure(fig, figure_name=figure_name)
        self._check_show()

        return fig, ax
