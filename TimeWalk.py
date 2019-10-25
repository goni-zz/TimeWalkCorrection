import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from itertools import groupby
from operator import itemgetter
from scipy import optimize
import os,sys

verbose = False
path = "inputHd5/"
filename = "laser.hd5"
outputD='results'
os.system('mkdir -p %s'%outputD)
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

#Normal distribution
from scipy.stats import norm
mu, std = norm.fit(data) #Gaussian fit
# Plot the histogram.
plt.hist(data, bins=20, density=True, alpha=0.6, color='b', label='t1-t0')
xmin, xmax = plt.xlim()
xnorm = np.linspace(xmin, xmax, 100)
p = norm.pdf(xnorm, mu, std)
#FWHM
from scipy.interpolate import UnivariateSpline
import pylab as pl
spline = UnivariateSpline(xnorm, p-np.max(p)/2, s=0)
fwhm = abs(spline.roots()[1]-spline.roots()[0])
print("Full Width Half Maximum:", fwhm)
pl.axvspan(spline.roots()[0], spline.roots()[1], facecolor='g', alpha=0.5, label='FWHM')
print("Sigma:",std)
norm.fit(data)
pl.plot(xnorm, p, c='r', label='Fit')
pl.legend(loc='best')
pl.savefig("%s/fig1_w_FWHM.pdf" %outputD)
pl.show()

plt.scatter(t1/dt,t0/dt)
plt.xlabel("t1 - Ch.2")
plt.ylabel("t0 - Ch.1")
plt.savefig("%s/fig_t1_vs_t0.pdf" %outputD)
##labels and save
plt.hist(data, bins=20, label='t1-t0')
plt.title("t1-t0 (1000evts)")
plt.xlabel("time(s)")
plt.ylabel("Events")
plt.legend(loc='best')
plt.savefig("%s/fig_wo_FWHM.pdf" %outputD)
#hypothesis tests null is normal dist. Alternate hypo is not null..
from scipy.stats import shapiro #shapiro
stat, p = shapiro(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
from statsmodels.graphics.gofplots import qqplot #qqplot quantile mapping with real data
qqplot(data, line='s')
plt.savefig("%s/hypothesis_tests.pdf" %outputD)

plt.figure(figsize=(12,8))
plt.subplot(211)
plt.scatter(x, y0, label='laser', color='r')
plt.plot(x, interp_fn0(x), color='black', label='interpolated by Newton Raphson method')
plt.grid(True)
plt.title("Ch1")
plt.ylabel("amplitude")
plt.hlines(y=y_t0, xmin=t0/dt-50, xmax=t0/dt+50, color='magenta', label='50% of maxV (laser)')
plt.vlines(x=t0_[999], ymin=-0.5, ymax=0.5, label='t0', color='g')
plt.legend(loc='best')

plt.subplot(212)
plt.scatter(x, y1, label='pre-amp pulse')
plt.plot(x, interp_fn1(x), color='black', label='interpolated by Newton Raphson method')
plt.grid(True)
plt.title("Ch2")
plt.xlabel("time")
plt.ylabel("amplitude")
plt.hlines(y=y_t1, xmin=t1/dt-50, xmax=t1/dt+50, label='20% of maxV (pre-amp)', color='b')
plt.vlines(x=t1_[999], ymin=-0.5, ymax=0.5, label='t1', color='purple')
plt.legend(loc='best')
plt.savefig("%s/figs.pdf" %outputD)
plt.show()
