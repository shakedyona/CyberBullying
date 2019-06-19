from .. import base_handler
from ... import api


class Train(base_handler.BaseHandler):
    def post(self):
        file = self.body_argument('path')
        api.train(file)
        self.set_status(200)

