from .. import base_handler
from ... import api
import os.path

SOURCE = os.path.abspath(os.path.join(__file__, '../../../'))


class Predict(base_handler.BaseHandler):
    def post(self):
        """
        HTTP POST method get 'post' and 'explainability' as body arguments
        'post' should be an Hebrew text (string) to classify
        'explainability' should be boolean where true is a request to include explanation of classification
        and false otherwise
        the request should be sent after 'train' occurs
        or if all the necessary models are saved in the 'output' directory
        return class: 1 if the post is offensive and class: 0 if its not offensive
        :return:
        """
        post = self.body_argument('post')
        explain = self.body_argument('explainability')

        data = api.get_classification(post, explain)
        if 'error' in data.keys():
            self.set_status(404)
            self.write(data)
            return

        self.set_status(200)
        self.write(data)


