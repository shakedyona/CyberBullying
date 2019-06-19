from .. import base_handler
from ... import api
import os.path

SOURCE = os.path.abspath(os.path.join(__file__, '../../../'))


class Predict(base_handler.BaseHandler):
    def post(self):
        post = self.body_argument('post')
        explain = self.body_argument('explainability')

        data = api.get_classification(post, explain)
        if 'error' in data.keys():
            self.set_status(404)
            self.write(data)
            return

        self.set_status(200)
        self.write(data)


