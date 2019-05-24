from .. import base_handler
from ... import api


class Train(base_handler.BaseHandler):
    def post(self):
        file = self.body_argument('file')
        succeed = api.train_file(file)
        if succeed:
            self.set_status(200)
        else:
            self.set_status(500)

