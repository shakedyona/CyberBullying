from .. import base_handler
from ... import api


class Train(base_handler.BaseHandler):
    def post(self):
        """
        HTTP POST method with path to csv file in the body request.
        the file should have 'text' column, 'id' column and 'cb_level' column.
        the models are trained with the file and return OK when finish.
        the trained models are saved in the "output" folder
        :return:
        """
        file = self.body_argument('path')
        api.train(file)
        self.set_status(200)

