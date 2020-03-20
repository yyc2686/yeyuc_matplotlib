import matplotlib.pyplot as plt
import numpy as np


class Template:

    def common_template(self, **kwargs):

        # 配置坐标轴与原点位置(spines): 是否使用笛卡尔坐标系
        if kwargs.get("cartesian"):
            # gca = 'get current axis'
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')

            ax.xaxis.set_ticks_position('bottom')
            # ACCEPTS: [ 'top' | 'bottom' | 'both' | 'default' | 'none' ]

            ax.spines['bottom'].set_position(('data', 0))
            # the 1st is in 'outward' | 'axes' | 'data'
            # axes: percentage of y axis
            # data: depend on y data

            ax.yaxis.set_ticks_position('left')
            # ACCEPTS: [ 'left' | 'right' | 'both' | 'default' | 'none' ]

            ax.spines['left'].set_position(('data', 0))

        # 设置图像有效范围(lim)
        self.axes.set_xlim(kwargs.get("xlim"))
        self.axes.set_ylim(kwargs.get("ylim"))
        if kwargs.get("zlim"):
            self.axes.set_zlim(kwargs.get("zlim"))

        # 设置坐标轴名称(label)
        if kwargs.get("xlabel"):
            self.axes.set_xlabel(kwargs.get("xlabel"))
        if kwargs.get("ylabel"):
            self.axes.set_ylabel(kwargs.get("ylabel"))

        # 设置坐标轴刻度(ticks)和标签(tick_labels)
        if type(kwargs.get("xticks")) == np.ndarray or kwargs.get(
                "xticks") == [] or kwargs.get("xticks"):
            self.axes.set_xticks(kwargs.get("xticks"))
        if type(kwargs.get("yticks")) == np.ndarray or kwargs.get(
                "yticks") == [] or kwargs.get("yticks"):
            self.axes.set_yticks(kwargs.get("yticks"))
        if kwargs.get("xtick_labels"):
            self.axes.set_xticklabels(kwargs.get("xtick_labels"))
        if kwargs.get("ytick_labels"):
            self.axes.set_yticklabels(kwargs.get("ytick_labels"))

        # 设置图例(legend)
        if kwargs.get("show_legend"):
            plt.legend(loc=kwargs.get("loc"))
        elif kwargs.get("legend_labels"):
            plt.legend(handles=self.handles[0]
            if len(self.handles) == 1 else self.handles,
                       labels=kwargs.get("legend_labels"),
                       loc=kwargs.get("loc", "best"))

            # the "," is very important in here l1, = plt... and l2, = plt... for this step
            """legend( handles=(line1, line2, line3),
                    labels=('label1', 'label2', 'label3'),
                    'upper right')
                The *loc* location codes are::

                    'best' : 0,          (currently not supported for figure legends)
                    'upper right'  : 1,
                    'upper left'   : 2,
                    'lower left'   : 3,
                    'lower right'  : 4,
                    'right'        : 5,
                    'center left'  : 6,
                    'center right' : 7,
                    'lower center' : 8,
                    'upper center' : 9,
                    'center'       : 10,"""

        # 设置标题
        if kwargs.get("title"):
            self.axes.set_title(kwargs.get("title"),
                                fontsize=12,
                                fontname="Times New Roman")

        # 对数坐标
        if kwargs.get("xlog"):
            self.axes.set_xscale('log')
        if kwargs.get("ylog"):
            self.axes.set_yscale('log')

        # # 设置坐标轴刻度的字体
        if kwargs.get("tick_font"):
            labels = self.axes.get_xticklabels() + self.axes.get_yticklabels()
            for label in labels:
                label.set_fontname('Times New Roman')
                label.set_fontsize(kwargs.get("tick_font"))
                label.set_bbox(
                    dict(facecolor=kwargs.get("facecolor", "white"),
                         edgecolor=kwargs.get("edgecolor", "none"),
                         alpha=kwargs.get("alpha", 0.8),
                         zorder=kwargs.get("zorder", 2)))

        # 设置色标(colorbar)
        if kwargs.get("colorbar"):
            plt.colorbar(shrink=kwargs.get("shrink", .92))
        return

    def subplots_example(self):
        """
        使用gridspec绘制多图
        """
        import matplotlib.gridspec as gridspec

        gs = gridspec.GridSpec(3, 3)
        # use index from 0
        ax1 = plt.subplot(gs[0, :])
        ax2 = plt.subplot(gs[1, :2])
        ax3 = plt.subplot(gs[1:, 2])
        ax4 = plt.subplot(gs[-1, 0])
        ax5 = plt.subplot(gs[-1, -2])
        return ax1, ax2, ax3, ax4, ax5
