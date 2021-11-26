import matplotlib.pyplot as plt
from cytoolz import pluck

a = [{1: 3}, {1: 4}]
print([*pluck(1,a)])
