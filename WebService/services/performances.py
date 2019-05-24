from .. import base_handler
from ... import api
import json


class Performances(base_handler.BaseHandler):
    def post(self):

        performances = api.get_performances()

        if performances:
            self.set_status(200)
            self.write(json.dumps(performances))
        else:
            self.set_status(500)
