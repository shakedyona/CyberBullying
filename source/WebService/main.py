# import argparse
import tornado.ioloop
import tornado.web
# import .configuration
from .services import train
from .services import predict
from .services import performances


def make_app(debug):
    return tornado.web.Application(
        [
            ('/train', train.Train),
            ('/GetClassification', predict.Predict),
            ('/GetPerformance', performances.Performances),
        ],
        debug=debug
    )


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config-file', default="/etc/rezilion/dashboard_backend.yaml")
    # parser.add_argument('--debug', action='store_true')
    # arguments = parser.parse_args()
    # config = rezilion_utils.configuration.load(arguments.config_file)
    # rezilion_utils.logs.initialize()
    app = make_app(True)              # arguments.debug
    app.listen(8888)  # Getting port number to listen to as an argument      config.webserver.port
    tornado.ioloop.IOLoop.current().start()
