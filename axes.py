import matplotlib.pyplot as plt
import numpy as np


class Axes:

    def draw_line(self, x, y, **kwargs):
        # the "," is very important in here l1, = plt... and l2, = plt... for this step
        l, = self.axes.plot(x,
                            y,
                            color=kwargs.get('color'),
                            marker=kwargs.get('marker'),
                            markersize=kwargs.get('markersize', 4),
                            markerfacecolor=kwargs.get('markerfacecolor'),
                            alpha=kwargs.get('alpha', 1),
                            linewidth=2,
                            linestyle='dashed')
        return l

    def draw_stackplot(self, x, y, **kwargs):
        s = self.axes.stackplot(x, y, baseline=kwargs.get("baseline", "zero"))
        return s

    def draw_scatter(self, x, y, **kwargs):
        s = self.axes.scatter(x,
                              y,
                              c=kwargs.get("c"),
                              s=kwargs.get("s"),
                              alpha=kwargs.get("alpha"))
        return s

    def draw_bar(self, x, y, **kwargs):
        b = self.axes.bar(x,
                          y,
                          width=kwargs.get("width", 0.8),
                          facecolor=kwargs.get("facecolor"),
                          edgecolor=kwargs.get("edgecolor"),
                          label=kwargs.get("label"))

        for x1, y1 in zip(x, y):
            # ha: horizontal alignment
            # va: vertical alignment
            self.axes.text(x1,
                           y1 + kwargs.get("ybias"),
                           '%.2f' % y1,
                           ha=kwargs.get("ha", "center"),
                           va=kwargs.get("va", "bottom"))
        return b

    def draw_barh(self, x, y, **kwargs):
        b = self.axes.barh(x,
                           y,
                           xerr=kwargs.get("xerr"),
                           error_kw=kwargs.get("error_kw", {
                               'ecolor': '0.1',
                               'capsize': 6
                           }),
                           color=kwargs.get("color"),
                           alpha=kwargs.get("alpha", 0.7),
                           label=kwargs.get("label"))
        return b

    def draw_barh1(self, x, y, **kwargs):
        b = self.axes.bar(x=0,
                          bottom=x,
                          width=y,
                          height=kwargs.get("height", 0.5),
                          color=kwargs.get("color"),
                          alpha=kwargs.get("alpha"),
                          orientation='horizontal',
                          label=kwargs.get("label"))
        return b

    def draw_pie(self, sizes, **kwargs):

        wedges, texts, autotexts = self.axes.pie(
            sizes,
            explode=kwargs.get("explode"),
            labels=kwargs.get("labels"),
            autopct=kwargs.get("autopct", '%1.1f%%'),
            shadow=kwargs.get("shadow", True),
            startangle=kwargs.get("startangle", 90))

        return wedges

    def draw_hist(self, x, bins, **kwargs):
        n, bins, patches = self.axes.hist(
            x=x,
            bins=bins,
            density=kwargs.get("density", True),
            histtype=kwargs.get("histtype", "bar"),
            cumulative=kwargs.get("cumulative"),
            label=kwargs.get("label"),
        )
        return n, bins, patches

    def draw_hist2d(self, x, y, bins, **kwargs):
        from matplotlib import colors

        h = self.axes.hist2d(
            x,
            y,
            bins=bins,
            norm=colors.LogNorm(),
        )
        return h

    def draw_hexbin(self, x, y, **kwargs):
        hb = self.axes.hexbin(x,
                              y,
                              gridsize=kwargs.get("gridsize", 50),
                              bins=kwargs.get("bins"),
                              cmap=kwargs.get("cmap", "inferno"))
        cb = plt.colorbar(hb, ax=self.axes)
        return hb

    def draw_errorbar(self, x, y, **kwargs):
        err = self.axes.errorbar(
            x,
            y,
            xerr=kwargs.get("xerr"),
            yerr=kwargs.get("yerr"),
            marker=kwargs.get("marker"),
            markersize=kwargs.get("markersize"),
            linestyle=kwargs.get("linestyle"),
        )
        return err

    def draw_boxplot(self, x, **kwargs):
        """
        x: data
        labels: labels of data
        showmeans: green triangle
        meanline: green line
        showbox: show box
        showcaps: show bottom and top
        notch: shape of "S"
        showfliers: show point

        boxprops: settings of box
        flierprops: settings of point
        medianprops: settings of mean
        meanprops: settings of mean
        meanlineprops: settings of mean

        # 常用属性配置
        boxprops = dict(linestyle='--', linewidth=3, color='darkgoldenrod')
        flierprops = dict(marker='o',
                        markerfacecolor='green',
                        markersize=12,
                        linestyle='none')
        medianprops = dict(linestyle='-.', linewidth=2.5, color='firebrick')
        meanpointprops = dict(marker='D',
                            markeredgecolor='black',
                            markerfacecolor='firebrick')
        meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')
        """
        box = self.axes.boxplot(
            x,
            labels=kwargs.get("labels"),
            showmeans=kwargs.get("showmeans"),
            meanline=kwargs.get("meanline"),
            showbox=kwargs.get("showbox", True),
            showcaps=kwargs.get("showcaps", True),
            notch=kwargs.get("notch"),
            bootstrap=kwargs.get("bootstrap"),
            showfliers=kwargs.get("showfliers", True),
            boxprops=kwargs.get("boxprops"),
            flierprops=kwargs.get("flierprops"),
            medianprops=kwargs.get("medianprops"),
            meanprops=kwargs.get("meanpointprops",
                                 kwargs.get("meanlineprops")),
        )
        return box

    def draw_violinplot(self, x, **kwargs):
        if kwargs.get("simple"):
            box = self.axes.violinplot(
                x,
                showmeans=kwargs.get("showmeans", True),
                showmedians=kwargs.get("meanline", True),
                showextrema=kwargs.get("showbox", True),
            )
        elif kwargs.get("complex"):

            def adjacent_values(vals, q1, q3):
                upper_adjacent_value = q3 + (q3 - q1) * 1.5
                upper_adjacent_value = np.clip(upper_adjacent_value, q3,
                                               vals[-1])

                lower_adjacent_value = q1 - (q3 - q1) * 1.5
                lower_adjacent_value = np.clip(lower_adjacent_value, vals[0],
                                               q1)
                return lower_adjacent_value, upper_adjacent_value

            box = self.axes.violinplot(
                x,
                showmeans=kwargs.get("showmeans", False),
                showmedians=kwargs.get("meanline", False),
                showextrema=kwargs.get("showbox", False),
            )

            for pc in box['bodies']:
                pc.set_facecolor('#D43F3A')
                pc.set_edgecolor('black')
                pc.set_alpha(1)

            quartile1, medians, quartile3 = np.percentile(x, [25, 50, 75],
                                                          axis=1)
            whiskers = np.array([
                adjacent_values(sorted_array, q1, q3)
                for sorted_array, q1, q3 in zip(x, quartile1, quartile3)
            ])
            whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

            inds = np.arange(1, len(medians) + 1)
            self.axes.scatter(inds,
                              medians,
                              marker='o',
                              color='white',
                              s=30,
                              zorder=3)
            self.axes.vlines(inds,
                             quartile1,
                             quartile3,
                             color='k',
                             linestyle='-',
                             lw=5)
            self.axes.vlines(inds,
                             whiskers_min,
                             whiskers_max,
                             color='k',
                             linestyle='-',
                             lw=1)
        return box

    def draw_contour(self, x, y, h, **kwargs):
        # use plt.contourf to filling contours
        # x, y and value for (x,y) point
        self.axes.contourf(x, y, h, 8, alpha=.75, cmap=plt.cm.hot)

        # use plt.contour to add contour lines
        C = self.axes.contour(x, y, h, 8, colors='black', linewidths=.5)
        # adding label
        self.axes.clabel(C, inline=True, fontsize=10)

    def draw_imshow(self, a, **kwargs):
        """
        for the value of "interpolation", check this:
        http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html
        for the value of "origin"= ['upper', 'lower'], check this:
        http://matplotlib.org/examples/pylab_examples/image_origin.html
        """
        im = plt.imshow(a,
                        interpolation=kwargs.get("interpolation", "nearest"),
                        cmap=kwargs.get("cmap", "bone"),
                        origin=kwargs.get("origin", "lower"))
        return im

    def draw_3D(self, x, y, z, **kwargs):
        from mpl_toolkits.mplot3d import Axes3D

        self.axes = Axes3D(fig=self.fig)
        self.axes.plot_surface(x,
                               y,
                               z,
                               rstride=1,
                               cstride=1,
                               cmap=plt.get_cmap('rainbow'))
        """
        ============= ================================================
        Argument      Description
        ============= ================================================
        *x*, *y*, *z* Data values as 2D arrays
        *rstride*     Array row stride (step size), defaults to 10
        *cstride*     Array column stride (step size), defaults to 10
        *color*       Color of the surface patches
        *cmap*        A colormap for the surface patches.
        *facecolors*  Face colors for the individual patches
        *norm*        An instance of Normalize to map values to colors
        *vmin*        Minimum value to map
        *vmax*        Maximum value to map
        *shade*       Whether to shade the facecolors
        ============= ================================================
        """

        # I think this is different from plt12_contours
        self.axes.contourf(x,
                           y,
                           z,
                           zdir='z',
                           offset=-2,
                           cmap=plt.get_cmap('rainbow'))
        """
        ==========  ================================================
        Argument    Description
        ==========  ================================================
        *x*, *y*,   Data values as numpy.arrays
        *z*
        *zdir*      The direction to use: x, y or z (default)
        *offset*    If specified plot a projection of the filled contour
                    on this position in plane normal to zdir
        ==========  ================================================
        """

        return
