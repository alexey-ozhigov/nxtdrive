#!/usr/bin/python

import matplotlib.pyplot as plt

data = [float(x) for x in file('dist_pixel.txt').readlines()]
plt.title('Range distribution of edge pixel')
plt.xlabel('Range, m')
plt.ylabel('Density')
plt.hist(data, bins=80)
plt.savefig('dist_pixel.png')
