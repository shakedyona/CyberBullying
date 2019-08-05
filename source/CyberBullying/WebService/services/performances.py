from .. import base_handler
from ... import api, utils
import json
import os.path
SOURCE = os.path.abspath(os.path.join(__file__, '../../../'))


class Performances(base_handler.BaseHandler):
    def post(self):
        """
        HTTP POST method with path to csv file in the body request.
        the file should have 'text' column, 'id' column and 'cb_level' column.
        return the precision, recall and f-measure and output images of the model as a base64 strings
        :return:
        """
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
