import os

import matplotlib.pyplot as plt


class Postprocessing:

    def fig_show(self):
        # plt.tight_layout()
        self.fig.show()
        # plt.pause(3)
        # plt.close()

    def fig_save(self, **kwargs):
        path = kwargs.get('path', './data')
        self.fig.savefig(os.path.join(path, kwargs.get('save_name') + '.pdf'),
                         dpi=500,
                         bbox_inches='tight')
