import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from itertools import groupby
from operator import itemgetter

verbose = False
path = "inputHd5/"
filename = "laser.hd5"

f = h5py.File(path+filename,'r')

dataset = f['C0']

#print(dataset.shape[0])
rng = f.attrs['Waveform Attributes'].tolist()
if verbose :
    print("What's inside in the 'waveform attributes': ")
    print("channel mask"+",", "number of points"+",", "number of frames"+",", u"\N{GREEK CAPITAL LETTER DELTA}"+"t"+",", "t0"+",", "y multiply values"+",", "y offsets"+",", "y zero")
    print(rng)

events = 1000
t0 = rng[4]
npoints = rng[1]
dt = rng[3]
ymult = 0.004

t1_ = np.repeat(None, events)

for iset in range(events):
    y_org = dataset[1, npoints * iset:npoints * (iset + 1)] * ymult

    y_t1 = min(y_org) + (max(y_org) - min(y_org)) * 0.8

    xmin = np.where(max(y_org) == y_org)[0][0]
    xmax = np.where(min(y_org) == y_org)[0][0]

    mask_ = np.where(y_org <= (y_t1 + 0.1))[0]
    y_ = y_org[mask_]
    x_ = np.arange(npoints)[mask_]
    y = y_[0:np.where(y_ == min(y_))[0][0]]
    x = x_[0:np.where(y_ == min(y_))[0][0]]

    xx = []
    yy = []
    for i, y in enumerate(y):
        if y not in yy:
            yy.append(y)
            xx.append(x[i])

    f = interp1d(yy, xx, kind='cubic')

    t1_[iset] = f(y_t1)

t1 = t1_*dt
plt.hist(t1-t0, bins=20)
plt.title("t1-t0 (1000evts)")
plt.xlabel("time(s)")
plt.ylabel("Events")

##### For plotting #####

iset = 600 #for random test whether the t1 and t0 are fine
y_org = dataset[1, npoints*iset:npoints*(iset+1)]*ymult
y_t1 = min(y_org)+(max(y_org)-min(y_org))*0.8

xmin = np.where(max(y_org)==y_org)[0][0]
xmax = np.where(min(y_org)==y_org)[0][0]

mask_ = np.where( y_org <= (y_t1+0.1))[0]
y_ = y_org[mask_]
x_ = np.arange(npoints)[mask_]
y = y_[0:np.where(y_ == min(y_))[0][0]]
x = x_[0:np.where(y_ == min(y_))[0][0]]

xx = [] # for duplicates of maximum values
yy = [] # for duplecates of maximum values
for i, y in enumerate(y):
    if y not in yy:
        yy.append(y)
        xx.append(x[i])

f = interp1d(yy, xx, kind='cubic') ## interpolate to fit between 2 points

plt.figure(figsize=(12,8))
plt.plot(np.arange(npoints), dataset[0, npoints:npoints*2]*ymult, label='input pulse', color='r')
plt.plot(np.arange(npoints), y_org, label='pre-amp')
plt.grid(True)
plt.title("TEST iset")
plt.xlabel("time(ps)")
plt.ylabel("amplitude")
plt.hlines(y=y_t1, xmin=700, xmax=900, label='20% of maxV (pre-amp)', color='b')
plt.vlines(x=f(y_t1), ymin=-0.5, ymax=0.5, label='t1', color='purple')
#plt.xlines(, xmin=-400, xmax=600, label='t0', color='g')
plt.vlines(x=t0/dt, ymin=-0.5, ymax=0.5, label='t0', color='g')
plt.legend(loc='best')
plt.show()
