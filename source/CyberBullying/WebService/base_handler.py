import json
import tornado.web
import tornado_cors


class BaseHandler(tornado_cors.CorsMixin, tornado.web.RequestHandler):
    """
    this class is the base of all the handlers in the services directory and which all should extend it.
    """
    CORS_ORIGIN = '*'
    CORS_HEADERS = 'Content-Type'
    CORS_METHODS = 'POST, GET'
    CORS_MAX_AGE = 21600
    CORS_EXPOSE_HEADERS = 'Location, X-WP-TotalPages'

    def write_error(self, status_code, **kwargs):
        """
        write an error as a response when occurs. send json with error code and error message
        :param status_code:
        :param kwargs:
        :return:
        """
        http_error = kwargs['exc_info'][1]
        error_log = ""
        if hasattr(http_error, 'log_message'):
            error_log = http_error.log_message
        self.set_status(status_code)
        self.write({'code': status_code, 'message': error_log})

    def prepare(self):
        """
        prepare called on each request and parse the body of the request into json object (dictionary)
        :return:
        """
        if len(self.request.body) > 0:
            self.request.json = json.loads(self.request.body)

    def body_argument(self, key):
        """
        get a specific key (argument) from the request. return 400 if the key is missing
        :param key:
        :return:
        """
        if key not in self.request.json:
            raise tornado.web.HTTPError(400, 'argument missing from request')
        return self.request.json[key]
