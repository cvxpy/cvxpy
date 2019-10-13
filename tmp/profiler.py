import cProfile

class Profiler(cProfile.Profile):
    def __init__(self):
        super()

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *excinfo):
        self.disable()
