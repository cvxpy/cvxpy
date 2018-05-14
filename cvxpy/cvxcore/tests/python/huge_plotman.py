from huge_testman import *
import matplotlib.pyplot as plt
import numpy as np

per_change = []
errorbars = []

for fname in oldtime:
	os = oldtime[fname]
	if len(os) == 0:
		continue
	ns = newtime[fname]
	all_per_changes = [(o - n) / o for o , n in zip(os, ns)]
	a = np.array(all_per_changes)
	per_change.append(np.percentile(a , 50))
	errorbars.append( [ np.percentile(a, 25), np.percentile(a, 75) ] ) 

for idx, f in enumerate(fnames):
	s = len(f) - f[::-1].find('/')
	e = len(f) - f[::-1].find('.') - 1
	fnames[idx] = f[s:e]

zfname = zip(per_change, fnames)
zfname.sort(reverse=True)

print zfname
# per_change = [ z[0]  for z in zfname ]
# fnames = [ z[1]  for z in zfname ]
# fig, ax = plt.subplots()
# spaces = [1.5 * i + .2 for i in range(len(per_change)) ]
# rects = ax.bar(spaces, per_change, .45, color = 'y', yerr=tuple(errorbars))

# def autolabel(rects, labels, heights):
# 	 for label, rect, height in zip(labels, rects, heights):
# 	 	 ax.text(rect.get_x()+rect.get_width()/2., max(1.07*height,0), label,\
# 	 	 	ha='center', va='bottom')



# autolabel(rects,fnames, per_change)
# ax.set_xticklabels( tuple( ['' for i in range(len(fnames))]))
# plt.ylabel('Percent speed up')
# plt.show()

