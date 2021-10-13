class MoveJointsException(Exception):
    def __init__(self, message="Failed to reach target joints"):
        self.message = message
        super().__init__(self.message)


class MoveJointsOutOfTimeException(MoveJointsException):
    def __init__(self, message="Not enough time to reach target joints"):
        self.message = message
        super().__init__(self.message)
