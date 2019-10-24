import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from itertools import groupby
from operator import itemgetter
from scipy import optimize

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

t0_ = np.repeat(None, events)
t1_ = np.repeat(None, events)

#for inverse function method
#for iset in range(events):
#    y_org = dataset[1, npoints * iset:npoints * (iset + 1)] * ymult
#
#    y_t1 = min(y_org) + (max(y_org) - min(y_org)) * 0.8
#
#    xmin = np.where(max(y_org) == y_org)[0][0]
#    xmax = np.where(min(y_org) == y_org)[0][0]
#
#    mask_ = np.where(y_org <= (y_t1 + 0.1))[0]
#    y_ = y_org[mask_]
#    x_ = np.arange(npoints)[mask_]
#    y = y_[0:np.where(y_ == min(y_))[0][0]]
#    x = x_[0:np.where(y_ == min(y_))[0][0]]
#
#    xx = []
#    yy = []
#    for i, y in enumerate(y):
#        if y not in yy:
#            yy.append(y)
#            xx.append(x[i])
#
#    f = interp1d(yy, xx, kind='cubic')
#
#    t1_[iset] = f(y_t1)
###
#for newton method
x = np.arange(npoints)
for iset in range(events):

    y0 = dataset[0, npoints*iset:npoints*(iset+1)]*ymult
    y1 = dataset[1, npoints*iset:npoints*(iset+1)]*ymult

    y_t0 = min(y0)+(max(y0)-min(y0))*0.5
    y_t1 = min(y1)+(max(y1)-min(y1))*0.8

    # interpolate
    interp_fn0 = InterpolatedUnivariateSpline(x, y0)
    interp_fn1 = InterpolatedUnivariateSpline(x, y1)
    #interp_fn = Rbf(x,y1) 

    t0_[iset] = optimize.newton(lambda x: interp_fn0(x)-y_t0, x0=510, maxiter=500)
    t1_[iset] = optimize.newton(lambda x: interp_fn1(x)-y_t1, x0=830, maxiter=500)

t0 = t0_*dt
t1 = t1_*dt
data = t1-t0

#print(t1)
#print(t0)
#print(t1-t0)
#sns.distplot(t1-t0, hist=True, kde=True,
#             color = 'darkblue',
#             hist_kws={'edgecolor':'black'},
#             kde_kws={'linewidth': 4})
#np.mean(t1-t0, axis=None, dtype=None, out=None, keepdims=False)

from scipy.stats import norm
mu, std = norm.fit(data)
# Plot the histogram.
plt.hist(data, bins=20, density=True, alpha=0.6, color='b')
xmin, xmax = plt.xlim()
xnorm = np.linspace(xmin, xmax, 100)
p = norm.pdf(xnorm, mu, std)


#FWHMx = np.linspace(10, 110, 1000)
#green = xnorm(FWHMx, 50, 10)
#pink = xnorm(FWHMx, 60, 10)
#spline = UnivariateSpline(data, xnorm-np.max(xnorm)/2, s=0)
#r1, r2 = spline.roots() # find the roots
from scipy.interpolate import UnivariateSpline
spline = UnivariateSpline(x,ynorm-50,s=0)
fwhm = abs(spline.roots()[1]-spline.roots()[0])

plt.plot(xnorm, p, c='r')
plt.hist(data, bins=20)
##labels and save
plt.title("t1-t0 (1000evts)")
plt.xlabel("time(s)")
plt.ylabel("Events")
plt.savefig("fig1.pdf")

#hypothesis tests null is normal dist. Alternate hypo is not null.. 
from scipy.stats import shapiro #shapiro
stat, p = shapiro(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
from statsmodels.graphics.gofplots import qqplot #qqplot quantile mapping with real data
qqplot(data, line='s')

##### For plotting #####
#iset = 600 #for random test whether the t1 and t0 are fine
#y_org = dataset[1, npoints*iset:npoints*(iset+1)]*ymult
#y_t1 = min(y_org)+(max(y_org)-min(y_org))*0.8
#
#xmin = np.where(max(y_org)==y_org)[0][0]
#xmax = np.where(min(y_org)==y_org)[0][0]
#
#mask_ = np.where( y_org <= (y_t1+0.1))[0]
#y_ = y_org[mask_]
#x_ = np.arange(npoints)[mask_]
#y = y_[0:np.where(y_ == min(y_))[0][0]]
#x = x_[0:np.where(y_ == min(y_))[0][0]]
#
#xx = [] # for duplicates of maximum values
#yy = [] # for duplecates of maximum values
#for i, y in enumerate(y):
#    if y not in yy:
#        yy.append(y)
#        xx.append(x[i])
#
#f = interp1d(yy, xx, kind='cubic') ## interpolate to fit between 2 points


plt.figure(figsize=(12,8))
plt.scatter(x, y0, label='input pulse', color='r')
plt.scatter(x, y1, label='pre-amp puls')
plt.plot(x, interp_fn0(x), color='black')
plt.plot(x, interp_fn1(x), color='black')
plt.grid(True)
plt.title("TEST iset")
plt.xlabel("time")
plt.ylabel("amplitude")
plt.hlines(y=y_t1, xmin=700, xmax=900, label='20% of maxV (pre-amp)', color='b')
plt.vlines(x=t1_[999], ymin=-0.5, ymax=0.5, label='t1', color='purple')
#plt.xlines(, xmin=-400, xmax=600, label='t0', color='g')
plt.vlines(x=t0/dt, ymin=-0.5, ymax=0.5, label='t0', color='g')
plt.legend(loc='best')
plt.savefig("fig2.pdf")

plt.show()
