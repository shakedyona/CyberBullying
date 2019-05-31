import base_handler
from source import api
import json


class Predict(base_handler.BaseHandler):
    def post(self):
        post = self.body_argument('post')
        explain = self.body_argument('explainability')

        data = api.predict(post, explain)

        if data:
            self.set_status(200)
            self.write(data)
        else:
            self.set_status(500)

