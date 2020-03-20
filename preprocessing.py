import numpy as np


class Preprocessing:

    def import_data(self, **kwargs):
        if kwargs.get("simple_line"):
            """
            绘制简单曲线
            """
            x = np.linspace(-10, 10, 50)
            y = 2 * x + 1
            return x, y
        elif kwargs.get("simple_lines"):
            """
            绘制多条简单曲线(在笛卡尔坐标系上)
            """
            x = np.linspace(-3, 3, 50)
            y1 = 2 * x + 1
            y2 = x ** 2
            return x, y1, y2
        elif kwargs.get("simple_scatter"):
            """
            绘制散点图
            """
            n = 1024  # data size
            x = np.random.normal(0, 1, n)
            y = np.random.normal(0, 1, n)
            t = np.arctan2(y, x)  # for color later on
            return x, y, t
        elif kwargs.get("simple_bars"):
            """
            绘制柱状图(Bar)
            """
            n = 12
            x = np.arange(n)
            y1 = (1 - x / float(n)) * np.random.uniform(0.5, 1.0, n)
            y2 = (1 - x / float(n)) * np.random.uniform(0.5, 1.0, n)
            return x, y1, y2
        elif kwargs.get("simple_contour"):
            """
            绘制等高线图
            """
            n = 256
            x = np.linspace(-3, 3, n)
            y = np.linspace(-3, 3, n)
            x, y = np.meshgrid(x, y)
            return x, y
        elif kwargs.get("simple_imshow"):
            """
            绘制图片(灰度矩阵)
            """
            a = np.array([
                0.313660827978, 0.365348418405, 0.423733120134, 0.365348418405,
                0.439599930621, 0.525083754405, 0.423733120134, 0.525083754405,
                0.651536351379
            ]).reshape(3, 3)
            return a
        elif kwargs.get("simple_3D"):
            """
            绘制3D图像
            """
            # x, y value
            x = np.arange(-4, 4, 0.25)
            y = np.arange(-4, 4, 0.25)
            x, y = np.meshgrid(x, y)
            r = np.sqrt(x ** 2 + y ** 2)
            # height value
            z = np.sin(r)
            return x, y, z
        elif kwargs.get("simple_subfigs"):
            """
            绘制画中画
            """
            x = [1, 2, 3, 4, 5, 6, 7]
            y = [1, 3, 4, 2, 5, 8, 6]
            return x, y
        elif kwargs.get("simple_twinx"):
            """
            绘制画中画
            """
            x = np.arange(0, 10, 0.1)
            y1 = 0.05 * x ** 2
            y2 = -1 * y1
            return x, y1, y2
