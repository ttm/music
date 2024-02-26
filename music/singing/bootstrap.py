import os
from .perform import sing

here = os.path.abspath(os.path.dirname(__file__))
ECANTORIXDIR = here + '/ecantorix'


def get_engine(method="http"):
    """pull ecantorix repo for local usage"""
    if method == "http":
        repo_url = 'https://github.com/ttm/ecantorix'
    elif method == "ssh":
        repo_url = 'git@github.com:ttm/ecantorix.git'
    else:
        raise ValueError('method not understood')
    os.system('git clone ' + repo_url + ' ' + ECANTORIXDIR)
    return


def make_test_song():
    t = 1
    t2 = .5
    t4 = .25
    text = "hey ma bro, why fly while dive?"
    notes = 7, 0, 5, 7, 11, 12, 7, 0
    durs = t2, t2, t4, t4, t, t4, t2
    sing(text, notes, durs)
