from werkzeug.wrappers import Response

class ContentAccessors:
    def xml(self): ...
    def lxml(self): ...
    def json(self): ...

class TestResponse(Response, ContentAccessors): ...
