
from cytoolz import keyfilter

def pick(whitelist, d):
    return keyfilter(lambda k: k in whitelist, d)

alphabet = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
print(pick(['a', 'b'], alphabet))