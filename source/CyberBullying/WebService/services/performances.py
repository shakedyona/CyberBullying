from .. import base_handler
from ... import api, utils
import json
import os.path
SOURCE = os.path.abspath(os.path.join(__file__, '../../../'))


class Performances(base_handler.BaseHandler):
    def post(self):
        file = self.body_argument('path')
        performances = api.get_performance(file)
        images = [utils.get_image_string(os.path.join(SOURCE, 'outputs/summary_plot_bar.png')),
                  utils.get_image_string(os.path.join(SOURCE, 'outputs/dependence_plot.png')),
                  utils.get_image_string(os.path.join(SOURCE, 'outputs/summary_plot.png'))]
        if performances:
            self.set_status(200)
            self.write(json.dumps({'performance': performances,
                                   'images': images}))
        else:
            self.set_status(500)
