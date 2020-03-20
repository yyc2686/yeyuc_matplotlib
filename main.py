import matplotlib.pyplot as plt
import numpy as np

from axes import Axes
from postprocessing import Postprocessing
from preprocessing import Preprocessing
from template import Template


class PythonMatplotlib(Preprocessing, Postprocessing, Template, Axes):

    def __init__(self, **kwargs):
        # 建立画板与画笔
        self.fig, self.axes = plt.subplots(num=kwargs.get("num"),
                                           figsize=kwargs.get("figsize"))
        self.handles = []

        # 常用配置
        self.ALPHA = [1, 1, 1, 1, 1, 1]
        self.COLOR = [plt.get_cmap('tab20c').colors[i] for i in [0, 4, 8, 12, 16, 18]]
        self.MARKER = ['^', 'o', 's', '*', '+', 'D']
        self.MARKER_COLOR = [plt.get_cmap('tab20c').colors[i] for i in [1, 5, 8, 12, 16, 18]]

    def simple_line(self, **kwargs):
        # 数据准备
        data = self.import_data(simple_line=True)

        # 绘制图形
        self.draw_line(*data, color='red', linewidth=1.0, linestyle='--')

        # 使用模板
        self.common_template(xlim=(-1, 2),
                             ylim=(-2, 3),
                             xlabel="I am x",
                             ylabel="I am y",
                             xticks=np.linspace(-1, 2, 5),
                             yticks=[-2, -1.8, -1, 1.22, 3],
                             ytick_labels=[
                                 r'$really\ bad$', r'$bad$', r'$normal$',
                                 r'$good$', r'$really\ good$'
                             ])

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_line"))

    def simple_lines(self, **kwargs):
        # 数据预处理
        x, y1, y2 = self.import_data(simple_lines=True)

        # 绘制图形
        self.handles.append(self.draw_line(x=x, y=y1))
        self.handles.append(
            self.draw_line(x=x,
                           y=y2,
                           color='red',
                           linewidth=1.0,
                           linestyle='--'))

        # 使用模板
        self.common_template(
            xlim=(-1, 2),
            ylim=(-2, 3),
            #  xlabel="I am x",
            #  ylabel="I am y",
            xticks=np.linspace(-1, 2, 5),
            yticks=[-2, -1.8, -1, 1.22, 3],
            ytick_labels=[
                r'$really\ bad$', r'$bad$', r'$normal$', r'$good$',
                r'$really\ good$'
            ],
            tick_font=12,
            cartesian=True,
            # legend_labels=["up", "down"]
        )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_lines"))

    def simple_scatters(self, **kwargs):
        # 数据预处理
        x, y, T = self.import_data(simple_scatter=True)

        # 绘制图形
        self.draw_scatter(x=x, y=y, s=75, c=T, alpha=0.5)

        # 使用模板
        self.common_template(
            xlim=(-1.5, 1.5),
            ylim=(-1.5, 1.5),
            xticks=[],
            yticks=[],
        )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_scatters"))

    def simple_bars(self, **kwargs):
        # 数据预处理
        x, y1, y2 = self.import_data(simple_bars=True)

        # 绘制图形
        self.handles.append(
            self.draw_bar(
                x=x,
                y=y1,
                ybias=0.05,
                facecolor='#9999ff',
                edgecolor='white',
            ))
        self.handles.append(
            self.draw_bar(x=x,
                          y=-y2,
                          ybias=-0.05,
                          facecolor='#ff9999',
                          edgecolor='white',
                          va="top"))

        # 使用模板
        self.common_template(
            xlim=(-.5, 12),
            ylim=(-1.25, 1.25),
            xticks=[],
            yticks=[],
        )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_bars"))

    def simple_contour(self, **kwargs):
        # 数据预处理

        def f(x, y):
            # the height function
            return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

        x, y = self.import_data(simple_contour=True)
        h = f(x, y)

        # 绘制图形
        self.draw_contour(x=x, y=y, h=h)

        # 使用模板
        self.common_template(
            xticks=[],
            yticks=[],
        )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_contour"))

    def simple_imshow(self, **kwargs):
        # 数据预处理
        a = self.import_data(simple_imshow=True)

        # 绘制图形
        self.handles.append(self.draw_imshow(a=a))

        # 使用模板
        self.common_template(
            xticks=[],
            yticks=[],
            colorbar=True,
        )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_imshow"))

    def simple_3D(self, **kwargs):
        # 数据预处理
        x, y, z = self.import_data(simple_3D=True)

        # 绘制图形
        self.handles.append(self.draw_3D(x=x, y=y, z=z))

        # 使用模板
        self.common_template(zlim=(-2, 2))

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_3D"))

    def simple_subplots(self, **kwargs):
        ax1, ax2, ax3, ax4, ax5 = self.subplots_example()

        self.axes = ax1
        self.simple_line(fig_show=False)

        self.axes = ax2
        self.simple_lines(fig_show=False)

        self.axes = ax3
        self.simple_scatters(fig_show=False)

        self.axes = ax4
        self.simple_bars(fig_show=False)

        self.axes = ax5
        self.simple_contour(fig_show=False)

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_subplots"))

    def simple_subfigs(self, **kwargs):

        self.common_template(xticks=[], yticks=[])  # 清楚原坐标值

        # 数据预处理
        x, y = self.import_data(simple_subfigs=True)

        # 绘制图形, 使用模板
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        self.axes = self.fig.add_axes([left, bottom, width,
                                       height])  # main axes
        self.draw_line(x=x, y=y, color='red')
        self.common_template(xlabel="x", ylabel="y", title="title")

        self.axes = self.fig.add_axes([0.2, 0.6, 0.25, 0.25])  # inside axes
        self.draw_line(x=y, y=x, color='blue')
        self.common_template(xlabel="x", ylabel="y", title="title inside 1")

        self.axes = self.fig.add_axes([0.6, 0.2, 0.25, 0.25])  # inside axes
        self.draw_line(x=y[::-1], y=x, color='green')
        self.common_template(xlabel="x", ylabel="y", title="title inside 2")

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_subfigs"))

    def simple_twinx(self, **kwargs):
        # self.common_template(xticks=[], yticks=[])  # 清楚原坐标值

        # 数据预处理
        x, y1, y2 = self.import_data(simple_twinx=True)

        # 绘制图形, 使用模板
        self.draw_line(x=x, y=y1, color='green')
        self.axes = self.axes.twinx()  # mirror the ax1
        self.draw_line(x=x, y=y2, color='blue')

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_twinx"))

    def stack_plot(self, **kwargs):
        """
        ==============
        Stackplot Demo
        ==============

        How to create stackplots with Matplotlib.

        Stackplots are generated by plotting different datasets vertically on
        top of one another rather than overlapping with one another. Below we
        show some examples to accomplish this with Matplotlib.
        """

        # 数据预处理
        x = [1, 2, 3, 4, 5]
        y1 = [1, 1, 2, 3, 5]
        y2 = [0, 4, 2, 6, 8]
        y3 = [1, 3, 5, 7, 9]
        y = np.vstack([y1, y2, y3])

        # 绘制图形
        self.handles.append(self.draw_stackplot(x=x, y=y))

        # 使用模板
        self.common_template(legend_labels=["Fibonacci ", "Evens", "Odds"],
                             loc="upper left")

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "stack_plot"))

    def simple_pie(self, **kwargs):
        """
        ===============
        Basic pie chart
        ===============

        Demo of a basic pie chart plus a few additional features.

        In addition to the basic pie chart, this demo shows a few optional features:

            * slice labels
            * auto-labeling the percentage
            * offsetting a slice with "explode"
            * drop-shadow
            * custom start angle

        Note about the custom start angle:

        The default ``startangle`` is 0, which would start the "Frogs" slice on the
        positive x-axis. This example sets ``startangle = 90`` such that everything is
        rotated counter-clockwise by 90 degrees, and the frog slice starts on the
        positive y-axis.
        """

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:

        # 数据预处理
        labels = ['Frogs', 'Hogs', 'Dogs', 'Logs']
        sizes = [15, 30, 45, 10]
        explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        # 绘制图形
        self.handles.append(self.draw_pie(sizes=sizes, explode=explode))
        # self.handles.append(
        #     self.draw_pie(sizes=sizes, labels=labels, explode=explode))

        self.axes.axis(
            'equal'
        )  # Equal aspect ratio ensures that pie is drawn as a circle.

        # 使用模板
        self.common_template(
            title="Matplotlib bakery: A pie",
            legend_labels=labels,
        )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_pie"))

    def simple_bars_combine(self, **kwargs):
        # 数据预处理
        labels = ['G1', 'G2', 'G3', 'G4', 'G5']
        x = np.arange(len(labels))
        y1 = [20, 34, 30, 35, 27]
        y2 = [25, 32, 34, 20, 25]
        width = 0.35

        # 绘制图形
        self.handles.append(
            self.draw_bar(x=x - (width + 0.01) / 2,
                          y=y1,
                          width=width,
                          ybias=0.05,
                          facecolor='#9999ff',
                          edgecolor='white',
                          label='Men'))

        self.handles.append(
            self.draw_bar(x=x + (width + 0.01) / 2,
                          y=y2,
                          width=width,
                          ybias=0.05,
                          facecolor='#ff9999',
                          edgecolor='white',
                          label='Women'))

        # 使用模板
        self.common_template(ylabel="Scores",
                             title="Scores by group and gender",
                             xticks=x,
                             xtick_labels=labels,
                             show_legend=True)

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name",
                                               "simple_bars_combine"))

    def horizontal_bar(self, **kwargs):
        # 数据预处理
        x = np.arange(5)
        labels = ['A', 'B', 'C', 'D', 'E']
        y = [5, 7, 3, 4, 6]
        std = [0.8, 1, 0.4, 0.9, 1.3]

        # 绘制图形
        if kwargs.get("fun1", True):
            self.handles.append(
                self.draw_barh(x=x, y=y, color='b', alpha=0.7, label='First'))
        else:
            self.handles.append(
                self.draw_barh1(x=x, y=y, alpha=0.7, color='b', label='First'))

        # 使用模板
        self.common_template(
            yticks=x,
            ytick_labels=labels,
            show_legend=True,
            loc=5,
            title="geek-docs.com",
        )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "horizontal_bar"))

    def simple_hist(self, **kwargs):
        """
        =========================================================
        Demo of the histogram (hist) function with a few features
        =========================================================

        In addition to the basic histogram, this demo shows a few optional features:

        * Setting the number of data bins.
        * The ``normed`` flag, which normalizes bin heights so that the integral of
        the histogram is 1. The resulting histogram is an approximation of the
        probability density function.
        * Setting the face color of the bars.
        * Setting the opacity (alpha value).

        Selecting different bin counts and sizes can significantly affect the shape
        of a histogram. The Astropy docs have a great section_ on how to select these
        parameters.

        .. _section: http://docs.astropy.org/en/stable/visualization/histogram.html
        """

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:

        # 数据预处理
        np.random.seed(19680801)

        # example data
        mu = 100  # mean of distribution
        sigma = 15  # standard deviation of distribution
        x = mu + sigma * np.random.randn(437)
        num_bins = 50

        # 绘制图形
        n, bins, patches = self.draw_hist(x=x, bins=num_bins)

        # add a 'best fit' line
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
             np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
        self.draw_line(x=bins, y=y, linestyle="--")

        # 使用模板
        self.common_template(
            xlabel="Smarts",
            ylabel="Probability density",
            title=r'Histogram of IQ: $\mu=100$, $\sigma=15$',
        )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name",
                                               "simple_histogram"))

    def cumulative_hist(self, **kwargs):
        """
        ==================================================
        Using histograms to plot a cumulative distribution
        ==================================================

        This shows how to plot a cumulative, normalized histogram as a
        step function in order to visualize the empirical cumulative
        distribution function (CDF) of a sample. We also show the theoretical CDF.

        A couple of other options to the ``hist`` function are demonstrated.
        Namely, we use the ``normed`` parameter to normalize the histogram and
        a couple of different options to the ``cumulative`` parameter.
        The ``normed`` parameter takes a boolean value. When ``True``, the bin
        heights are scaled such that the total area of the histogram is 1. The
        ``cumulative`` kwarg is a little more nuanced. Like ``normed``, you
        can pass it True or False, but you can also pass it -1 to reverse the
        distribution.

        Since we're showing a normalized and cumulative histogram, these curves
        are effectively the cumulative distribution functions (CDFs) of the
        samples. In engineering, empirical CDFs are sometimes called
        "non-exceedance" curves. In other words, you can look at the
        y-value for a given-x-value to get the probability of and observation
        from the sample not exceeding that x-value. For example, the value of
        225 on the x-axis corresponds to about 0.85 on the y-axis, so there's an
        85% chance that an observation in the sample does not exceed 225.
        Conversely, setting, ``cumulative`` to -1 as is done in the
        last series for this example, creates a "exceedance" curve.

        Selecting different bin counts and sizes can significantly affect the
        shape of a histogram. The Astropy docs have a great section on how to
        select these parameters:
        http://docs.astropy.org/en/stable/visualization/histogram.html

        """

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:

        # 数据预处理
        mu = 200
        sigma = 25
        num_bins = 50
        x = np.random.normal(mu, sigma, size=100)

        # 绘制图形
        n, bins, patches = self.draw_hist(x=x,
                                          bins=num_bins,
                                          density=True,
                                          histtype="step",
                                          cumulative=True,
                                          label='Empirical')

        # add a 'best fit' line
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
             np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
        y = y.cumsum()
        y /= y[-1]
        self.draw_line(x=bins, y=y, linestyle="--")

        # 使用模板
        self.common_template(
            xlabel="Annual rainfall (mm)",
            ylabel="Likelihood of occurrence",
            title="Cumulative step histograms",
            show_legend=True,
            loc="right",
        )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "cumulative_hist"))

    def hist_2d(self, **kwargs):
        # Fixing random state for reproducibility
        np.random.seed(19680801)

        ###############################################################################
        # Generate data and plot a simple histogram
        # -----------------------------------------
        #
        # To generate a 1D histogram we only need a single vector of numbers. For a 2D
        # histogram we'll need a second vector. We'll generate both below, and show
        # the histogram for each vector.

        # 数据预处理
        # Generate a normal distribution, center at x=0 and y=5
        x = np.random.randn(100000)
        y = .4 * x + np.random.randn(100000) + 5
        # n_bins = (20, 20)
        n_bins = 40

        # 绘制图形
        self.handles.append(self.draw_hist2d(x=x, y=y, bins=n_bins))

        # 使用模板
        self.common_template()

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "hist_2d"))

    def simple_hexbin(self, **kwargs):
        """
        ===========
        Hexbin Demo
        ===========

        Plotting hexbins with Matplotlib.

        Hexbin is an axes method or pyplot function that is essentially
        a pcolor of a 2-D histogram with hexagonal cells.  It can be
        much more informative than a scatter plot. In the first plot
        below, try substituting 'scatter' for 'hexbin'.
        """

        # 数据预处理
        # Fixing random state for reproducibility
        np.random.seed(19680801)

        n = 100000
        x = np.random.standard_normal(n)
        y = 2.0 + 3.0 * x + 4.0 * np.random.standard_normal(n)
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()

        # 绘制图形
        self.handles.append(
            self.draw_hexbin(x=x, y=y, gridsize=50, cmap='inferno',
                             bins="log"))

        # 使用模板
        self.common_template(xlim=(xmin, xmax), ylim=(ymin, ymax))

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "hist_2d"))

    def simple_boxplot(self, **kwargs):
        """
        =================================
        Artist customization in box plots
        =================================

        This example demonstrates how to use the various kwargs
        to fully customize box plots. The first figure demonstrates
        how to remove and add individual components (note that the
        mean is the only value not shown by default). The second
        figure demonstrates how the styles of the artists can
        be customized. It also demonstrates how to set the limit
        of the whiskers to specific percentiles (lower right axes)

        A good general reference on boxplots and their history can be found
        here: http://vita.had.co.nz/papers/boxplots.pdf

        """

        # 数据预处理
        np.random.seed(19680801)
        data = np.random.lognormal(size=(37, 4), mean=1.5, sigma=1.75)
        labels = list('ABCD')
        fs = 10  # fontsize

        # 绘制图形
        self.handles.append(
            self.draw_boxplot(x=data,
                              labels=labels,
                              showmeans=True,
                              meanline=True,
                              notch=True,
                              bootstrap=10000,
                              showfliers=False))

        # 使用模板
        self.common_template(title="simple_boxplot", )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_boxplot"))

    def simple_violinplot(self, **kwargs):
        """
        =========================
        Violin plot customization
        =========================

        This example demonstrates how to fully customize violin plots.
        The first plot shows the default style by providing only
        the data. The second plot first limits what matplotlib draws
        with additional kwargs. Then a simplified representation of
        a box plot is drawn on top. Lastly, the styles of the artists
        of the violins are modified.

        For more information on violin plots, the scikit-learn docs have a great
        section: http://scikit-learn.org/stable/modules/density.html
        """

        # 数据预处理
        np.random.seed(19680801)
        data = [sorted(np.random.normal(0, std, 100)) for std in range(1, 5)]
        labels = ['A', 'B', 'C', 'D']

        # 绘制图形
        if kwargs.get("simple"):
            self.handles.append(self.draw_violinplot(simple=True, x=data))
        elif kwargs.get("complex"):
            self.handles.append(self.draw_violinplot(complex=True, x=data))

        # 使用模板
        self.common_template(title="Customized violin plot",
                             ylabel="Observed values",
                             xticks=np.arange(1,
                                              len(labels) + 1),
                             xtick_labels=labels)

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name",
                                               "simple_violinplot"))

    def simple_heatmap(self, **kwargs):
        # 数据预处理

        x = np.random.rand(100).reshape(10, 10)
        # x = np.random.rand(16).reshape(4, 4)
        # import pandas as pd
        # attr = ['a', 'b', 'c', 'd']
        # x = pd.DataFrame(x, columns=attr, index=attr)

        # 绘制图形, 使用模板
        if kwargs.get("imshow"):
            plt.imshow(x, cmap=plt.cm.hot, vmin=0, vmax=1)
            save_name = "simple_heatmap_imshow"
            self.common_template(colorbar=True)
        elif kwargs.get("matshow"):
            plt.matshow(x, cmap=plt.cm.cool, vmin=0, vmax=1)
            save_name = "simple_heatmap_matshow"
            self.common_template(colorbar=True)
        elif kwargs.get("seaborn"):
            import seaborn as sns
            sns.heatmap(x, vmin=0, vmax=1, center=0)
            save_name = "simple_heatmap_seaborn"

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", save_name))


if __name__ == '__main__':
    client = PythonMatplotlib()
    client.simple_line(save_as_pdf=True)
    client.fig, client.axes = plt.subplots()

    client.simple_lines(save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.simple_scatters(save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.simple_bars(save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.simple_contour(save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.simple_imshow(save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.simple_3D(save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.simple_subplots(save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.simple_subfigs(save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.simple_twinx(save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.stack_plot(save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.simple_pie(save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.simple_bars_combine(save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.horizontal_bar(save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.simple_hist(save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.cumulative_hist(save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.hist_2d(save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.simple_hexbin(save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.simple_boxplot(save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.simple_violinplot(simple=True, save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.simple_violinplot(complex=True, save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.simple_heatmap(imshow=True, save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.simple_heatmap(matshow=True, save_as_pdf=False)
    client.fig, client.axes = plt.subplots()

    client.simple_heatmap(seaborn=True, save_as_pdf=False)
    client.fig, client.axes = plt.subplots()
    pass
