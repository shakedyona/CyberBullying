import base_handler
from source import api


class Train(base_handler.BaseHandler):
    def post(self):
        file = self.body_argument('file')
        api.train(file)
        self.set_status(200)

