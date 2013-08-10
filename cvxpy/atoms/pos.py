from max import max

class pos(max):
    """ Elementwise max{x,0}. """
    def __init__(self, x):
        super(pos, self).__init__(x,0)