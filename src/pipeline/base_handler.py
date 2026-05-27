class BaseHandler:

    def __init__(self):
        self.next = None

    def set_next(self, handler):
        self.next = handler
        return handler

    def handle(self, data):
        if self.next:
            return self.next.handle(data)
        return data