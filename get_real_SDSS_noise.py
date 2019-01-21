from astropy.io import fits
from astropy.table import Table
from astropy.nddata.utils import Cutout2D
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

import numpy as np

import pickle

import matplotlib.pyplot as plt


#Set identifier number so noise names are unique
#idee = 752*243
#idee = 756*429
idee = 6793*130

#Image path and file names
img_path = "/Volumes/Data_Disk/sky_maps/sdss/dr7/"
#img_file_g = "fpC-000752-g4-0243.fit"
#img_file_r = "fpC-000752-r4-0243.fit.gz"
#img_file_i = "fpC-000752-i4-0243.fit.gz"
#img_file_g = "fpC-000756-g4-0429.fit.gz"
#img_file_r = "fpC-000756-r4-0429.fit.gz"
#img_file_i = "fpC-000756-i4-0429.fit.gz"
img_file_g = "fpC-006793-g4-0130.fit.gz"
img_file_r = "fpC-006793-r4-0130.fit.gz"
img_file_i = "fpC-006793-i4-0130.fit.gz"

#Catalogue path and names for positions of objects in each image
cat_path = "/Volumes/Data_Disk/catalogues/sdss/dr7/"
#cat_file = "radec-000752-4-40-0243.fit"
#tsF_file = "tsField-000752-4-40-0243.fit"
#cat_file = "radec-000756-4-44-0429.fit"
#tsF_file = "tsField-000756-4-44-0429.fit"
cat_file = "radec-006793-4-40-0130.fit"
tsF_file = "tsField-006793-4-40-0130.fit"

#Path to save noise
out_path = "/Volumes/Data_Disk/sky_maps/sdss/noise/"


#Load data
data = fits.getdata(cat_path+tsF_file, 0)
tsF = Table(data)
aa = np.array(tsF['aa'][0])
print(aa, '\n')
kk = np.array(tsF['kk'][0])
print(kk, '\n')
airmass = np.array(tsF['airmass'][0])
print(airmass)
tsF = []

#Load object positions
data = fits.getdata(cat_path+cat_file, 1)
table = Table(data)


#Open images in gri
fpC_g = fits.open(img_path+img_file_g)
exptime_g = fpC_g[0].header['EXPTIME']
exptime_g = float(exptime_g.strip("=' /Exposuretim(cnd)"))
softbias_g = fpC_g[0].header['SOFTBIAS']
img_g = fpC_g[0].data - softbias_g #Correct for count offset
wcs_g = WCS(fpC_g[0].header)

fpC_r = fits.open(img_path+img_file_r)
exptime_r = fpC_r[0].header['EXPTIME']
exptime_r = float(exptime_r.strip("=' /Exposuretim(cnd)"))
softbias_r = fpC_r[0].header['SOFTBIAS']
img_r = fpC_r[0].data - softbias_r
wcs_r = WCS(fpC_r[0].header)

fpC_i = fits.open(img_path+img_file_i)
exptime_i = fpC_i[0].header['EXPTIME']
exptime_i = float(exptime_i.strip("=' /Exposuretim(cnd)"))
softbias_i = fpC_i[0].header['SOFTBIAS']
img_i = fpC_i[0].data - softbias_i
wcs_i = WCS(fpC_i[0].header)

#Convert DN to flux units
img_g = (img_g / exptime_g) * np.power(10., 0.4*(aa[1]+(kk[1]*airmass[1])) ) #Convert to f_f0
img_r = (img_r / exptime_r) * np.power(10., 0.4*(aa[2]+(kk[2]*airmass[2])) )
img_i = (img_i / exptime_i) * np.power(10., 0.4*(aa[3]+(kk[3]*airmass[3])) )

img_g *= 3.631e6 #Convert to mJy
img_r *= 3.631e6
img_i *= 3.631e6

failed = 0

#Get noise
for i in range(0, len(table)):
    if i % (len(table)//10) == 0:
        print(i, 'of', len(table))
    #Move centre off object
    offset = 1.758e-3 + (np.random.uniform()*3.516e-3)
    angle = np.random.randint(0, 360)
    dRA = offset*np.sin(angle)
    dDEC  =offset*np.cos(angle)
    RA = table['ra'][i] + dRA
    DEC = table['dec'][i] + dDEC
    
    pos = SkyCoord(RA*u.deg, DEC*u.deg)
    okay = True
    #Get cutouts and ensure data exists in all three channels
    try:
        cutout_g = Cutout2D(img_g, pos, 256, wcs=wcs_g, mode='strict')
    except:
        okay = False
    else:
        try:
            cutout_r = Cutout2D(img_r, pos, 256, wcs=wcs_r, mode='strict')
        except:
            okay = False
        else:
            try:
                cutout_i = Cutout2D(img_i, pos, 256, wcs=wcs_i, mode='strict')
            except:
                okay = False
    #If data is not in all three channels, move on
    if not okay:
        failed += 1
        continue
    
    #Make 3 channel image..
    S = np.dstack((cutout_g.data, cutout_r.data, cutout_i.data))
    #...and save as pickle as we want the flux values
    with open(out_path+str(idee+i)+'.pkl', "wb") as f:
        pickle.dump(S, f)
        f.close()

#Show an example
if okay:
    print(RA, DEC)
    print(idee+i)
    plt.imshow(S/np.max(S))
    plt.show()
#Show how many noise cutouts have been made
print('Created', len(table)-failed, 'of', len(table))

