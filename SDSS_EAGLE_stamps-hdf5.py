import numpy as np
from PIL import Image

import h5py
from astropy.io import fits
from astropy.table import Table
from astropy.convolution import convolve
from astropy.cosmology import WMAP7 as cosmo
import astropy.units as u
from astropy import wcs

import glob
import pickle
import os

import matplotlib.pyplot as plt

#Set up data paths
path = '/Volumes/Data_Disk/sky_maps/eagle/no-dust/'
division = ['premerger', 'postmerger', 'nomerger'] #'premerger', 'postmerger', 'merge', 'nomerger'
no_dim = len(division)
extn = '.hdf5'

path_psf = '/Volumes/Data_Disk/sky_maps/sdss/eagle-nodust2-'
redshift_file = '/Volumes/Data_Disk/catalogues/sdss/darg_2010_mergers.fits'
z_col = 'specz1'
noise_path = '/Volumes/Data_Disk/sky_maps/sdss/noise/'

do_redshift = True   #Correct surface brightness for redshift
do_reproject = True  #Match physical resolution with angular resolution at redshift
do_convolve = True   #Convolve image with SDSS PSF
do_noise = True      #Add real SDSS noise

high_z = False #Only use redshifts above ~0.03 so the simulated image is always made smaller when reprojected

#Name output file to match what has been done
if do_redshift:
    path_psf += 'z'#'-do_redshift'
if do_reproject:
    path_psf += 'r'#'-do_reproject'
if do_convolve:
    path_psf += 'c'#'-do_convolve'
if do_noise:
    path_psf += 'n'#'-do_noise'
path_psf += '/'

#Channels and projections to use
ch = np.array(['i', 'r', 'g']) #gri colours (but it's irg to fit rgb)
projection = np.array(['_00', '_01', '_02', '_03', '_04', '_05'])

scale = 1.0
d_eagle = 10e-6 #Distance, in Mpc, that the eagle objects are

px = 1.0 #Not really needed. It's just a placeholder that's needed

#Plot example images
plot = False


#Make output folders (if needed)
if not os.path.exists(path_psf):
    os.makedirs(path_psf)
for div in division:
    if not os.path.exists(path_psf+div):
        os.makedirs(path_psf+div)


#Get dict of all EAGLE objects to make stamps
objs = {}
for div in division:
    objs[div] = glob.glob(path+div+'/*'+extn)

noise = glob.glob(noise_path+'*.pkl')


#Load PSF
psf_scale = 1.0

psf_g = fits.getdata('/Volumes/Data_Disk/sky_maps/sdss/psf/sdss_psf_g.fits', 0)
psf_g = np.array(psf_g, dtype='float')
psf_g -= 1000 #Remove 0pt bias
psf_g /= psf_scale
print(np.max(psf_g), np.min(psf_g), np.sum(psf_g))
plt.imshow(psf_g)
plt.show()

psf_r = fits.getdata('/Volumes/Data_Disk/sky_maps/sdss/psf/sdss_psf_r.fits', 0)
psf_r = np.array(psf_r, dtype='float')
psf_r -= 1000
psf_r /= psf_scale
print(np.max(psf_r), np.min(psf_r), np.sum(psf_r))
plt.imshow(psf_r)
plt.show()

psf_i = fits.getdata('/Volumes/Data_Disk/sky_maps/sdss/psf/sdss_psf_i.fits', 0)
psf_i = np.array(psf_i, dtype='float')
psf_i -= 1000
psf_i /= psf_scale
print(np.max(psf_i), np.min(psf_i), np.sum(psf_i))
plt.imshow(psf_i)
plt.show()

#Load observation data (for redshifts)
data = fits.getdata(redshift_file, 1)
table = Table(data)

from reproject import reproject_interp as repro


def img_func(I, beta=1.0, up=None, lo=0.0):
    if up == None:
        up = np.max(i)
    fI = np.arcsinh(I/beta)/np.arcsinh((up-lo)/beta)
    bad = np.where(I < lo)
    fI[bad] = 0.0
    bad = np.where(i > up)
    fI[bad] = 1.0
    return fI


#Scale to SDSS resolution
blank = np.zeros((256,256))

#WCS for new projection
wcs_new = wcs.WCS(naxis=2)
wcs_new.wcs.crpix = [blank.shape[0]//2, blank.shape[1]//2]
wcs_new.wcs.cdelt = np.array([0.00011, 0.00011])
wcs_new.wcs.crval = [0., 0.]
wcs_new.wcs.ctype = ["RA---TAN", "DEC--TAN"]
#WCS for EAGLE projection
wcs_egl = wcs.WCS(naxis=2)
wcs_egl.wcs.crpix = [blank.shape[0]//2, blank.shape[1]//2]
wcs_egl.wcs.cdelt = np.array([px/3600, px/3600])
wcs_egl.wcs.crval = [0., 0.]
wcs_egl.wcs.ctype = ["RA---TAN", "DEC--TAN"]

#Loop over the divisions/classes
for div in division:
    print(div)
    tenpct = len(objs[div])//10
    
    #Store redshifts
    obj_name = []
    obj_zsft = []
    
    #Loop over objects in div
    for i in range(0, len(objs[div])):
        if plot:
            i = np.random.randint(len(objs[div]))
            print(objs[div][i])
        if i % tenpct == 0:
            print(i, 'of', len(objs[div]))
        
        #Loop over the simulation projections
        for p in projection:
            #Load data, has to be reloaded for reasons (IDK)
            f = h5py.File(objs[div][i], 'r')
            
            px = 1.0
            #get redshift and determine effective resolution of EAGLE
            if high_z:
                while px/3600 > 0.00011:
                    rand_idx = np.random.randint(len(table))
                    kpc_as = cosmo.kpc_proper_per_arcmin(table[z_col][rand_idx]).to(u.kpc/u.arcsec)
                    px = ( ((60*u.kpc) / kpc_as)/256 ).value #calculate pixel size at redshift
            else:
                rand_idx = np.random.randint(len(table))
                kpc_as = cosmo.kpc_proper_per_arcmin(table[z_col][rand_idx]).to(u.kpc/u.arcsec)
                px = ( ((60*u.kpc) / kpc_as)/256 ).value #calculate pixel size at redshift
            wcs_egl.wcs.cdelt = np.array([px/3600, px/3600]) #update eagle WCS
            
            obj_name.append(objs[div][i][len(path)+len(div)+1:-len(extn)]+p)
            obj_zsft.append(table[z_col][rand_idx])
            
            if do_redshift:
                r = np.array(f[ch[0]+p])  #Load data
                g = np.array(f[ch[1]+p])
                b = np.array(f[ch[2]+p])
            
                if plot:
                    print('Origional')
                    print(np.max(np.dstack((r, g, b))))
                    plt.imshow(np.dstack((r[96:160,96:160]/np.max(r[96:160,96:160]),
                                          g[96:160,96:160]/np.max(g[96:160,96:160]),
                                          b[96:160,96:160]/np.max(b[96:160,96:160]))))
                    plt.show()
                    print('Pixel size of', round(px,3), ' arcsec at z =', table[z_col][rand_idx])
                #correct brightness for distance
                d_redsh = cosmo.luminosity_distance(table[z_col][rand_idx]).value
                r *= (d_eagle*d_eagle)/(d_redsh*d_redsh)
                g *= (d_eagle*d_eagle)/(d_redsh*d_redsh)
                b *= (d_eagle*d_eagle)/(d_redsh*d_redsh)
                #Convert from maggies to mJy (do now to prevent overflow)
                r *= 3.631e6
                g *= 3.631e6
                b *= 3.631e6
                #Scale to match g-band observations
                r *= scale
                g *= scale
                b *= scale
            else:
                r = np.array(f[ch[0]+p])  #Load data
                g = np.array(f[ch[1]+p])
                b = np.array(f[ch[2]+p])
                r *= 3.631e6    #Convert from maggies to mJy
                g *= 3.631e6
                b *= 3.631e6
                r *= scale      #Scale to match g-band observations
                g *= scale
                b *= scale

                if plot:
                    print('Origional')
                    print(np.max(np.dstack((r, g, b))))
                    plt.imshow(np.dstack((r[96:160,96:160]/np.max(r[96:160,96:160]),
                                          g[96:160,96:160]/np.max(g[96:160,96:160]),
                                          b[96:160,96:160]/np.max(b[96:160,96:160]))))
                    plt.show()

            #Reproject
            if do_reproject:
                #reproject from EAGLE to new WCS
                r, _ = repro((r,wcs_egl), wcs_new, shape_out=(256,256))
                r = r[96:160,96:160]        #Crop to center 64 pixels
                bad = np.where(np.isnan(r)) #Find nans...
                r[bad] = 0.0                #...and replace with zeros

                g, _ = repro((g,wcs_egl), wcs_new, shape_out=(256,256))
                g = g[96:160,96:160]
                bad = np.where(np.isnan(g))
                g[bad] = 0.0

                b, _ = repro((b,wcs_egl), wcs_new, shape_out=(256,256))
                b = b[96:160,96:160]
                bad = np.where(np.isnan(b))
                b[bad] = 0.0

                if plot:
                    print('Reprojected')
                    print(np.max(np.dstack((r, g, b))))
                    plt.imshow(np.dstack((r/np.max(r),
                                          g/np.max(g),
                                          b/np.max(b))))
                    plt.show()
            else:
                r = r[96:160,96:160] #If not reprojecting, just crop to center 64 pixels
                g = g[96:160,96:160]
                b = b[96:160,96:160]

            #Convolve with PSF
            if do_convolve:
                #Convolve EAGLE image with SDSS PSF
                r = convolve(r, psf_i)
                g = convolve(g, psf_r)
                b = convolve(b, psf_g)

                if plot:
                    print('Convolved')
                    print(np.max(np.dstack((r, g, b))))
                    plt.imshow(np.dstack((r/np.max(r),
                                          g/np.max(g),
                                          b/np.max(b))))
                    plt.show()

            #Add noise
            if do_noise:
                noise_idx = np.random.randint(0, len(noise)) #Pick random noise
                with open(noise[noise_idx], "rb") as f:
                    noise_map = pickle.load(f)
                    f.close()
                rot = np.random.randint(4, size=1)   #Pick random 90 degree rotation

                if plot:
                    print('Noise:', noise[noise_idx])
                    print(np.max(noise_map[96:160,96:160]), np.mean(noise_map[96:160,96:160]), np.min(noise_map[96:160,96:160]))
                    plt.imshow(noise_map[96:160,96:160]/np.max(noise_map[96:160,96:160]))
                    plt.show()
                    print('Histograms:')
                    bins = np.arange(-5, -2, 0.1)
                    plt.figure(figsize=(1.2*5.5, 3*1.25*(5.5*2)/3.0))
                    #print('\tr')
                    plt.subplot(3, 1, 1)
                    plt.title('r')
                    plt.hist(np.log10(noise_map[96:160,96:160,2].flatten()), label='Noise', alpha=0.5, bins=bins)
                    plt.hist(np.log10(r.flatten()), label='Sim', alpha=0.5, log=True, bins=bins)
                    #print('\tg')
                    plt.subplot(3, 1, 2)
                    plt.title('g')
                    plt.hist(np.log10(noise_map[96:160,96:160,1].flatten()), label='Noise', alpha=0.5, bins=bins)
                    plt.hist(np.log10(g.flatten()), label='Sim', alpha=0.5, log=True, bins=bins)
                    #print('\tb')
                    plt.subplot(3, 1, 3)
                    plt.title('b')
                    plt.hist(np.log10(noise_map[96:160,96:160,0].flatten()), label='Noise', alpha=0.5, bins=bins)
                    plt.hist(np.log10(b.flatten()), label='Sim', alpha=0.5, log=True, bins=bins)

                r += np.rot90(noise_map[96:160,96:160,2], rot) #Add noise, rotated by random 90 degree, to EAGLE
                g += np.rot90(noise_map[96:160,96:160,1], rot)
                b += np.rot90(noise_map[96:160,96:160,0], rot)

                if plot:
                    plt.subplot(3, 1, 1)
                    plt.hist(np.log10(r.flatten()), label='Combined', alpha=0.5, log=True, bins=bins)
                    plt.legend(loc=2)
                    plt.subplot(3, 1, 2)
                    plt.hist(np.log10(g.flatten()), label='Combined', alpha=0.5, log=True, bins=bins)
                    plt.legend(loc=2)
                    plt.subplot(3, 1, 3)
                    plt.hist(np.log10(b.flatten()), label='Combined', alpha=0.5, log=True, bins=bins)
                    plt.legend(loc=2)
                    plt.show()
                    print('With Noise')
            
            #Lupton et al. 2004, 2004PASP..116..133L
            I = (r+g+b)/3 #Define I
            up = np.min([np.max(r), np.max(g), np.max(b)]) #get upper limit
            #Zero correct colour channels
            lo = np.min([np.min(r), np.min(g), np.min(b)])
            r -= np.min(r)#lo
            g -= np.min(g)#lo
            b -= np.min(b)#lo
            #Define big RGB
            R = r*img_func(I, beta=1., up=up)/I
            G = g*img_func(I, beta=1., up=up)/I
            B = b*img_func(I, beta=1., up=up)/I
            #zero out the bootm
            zro = np.where(I == 0)
            R[zro] = 0.0
            G[zro] = 0.0
            B[zro] = 0.0
            #One out the Top
            RGB = np.dstack((R,G,B))
            one = np.where(RGB > 1)
            RGB[one] = 1.0
            #Convert to Image
            RGB = RGB * 255
            #print(np.max(RGB), np.min(RGB))
            IMG = Image.fromarray(RGB.astype(np.uint8), mode='RGB')
            
            #Linear scaling
            rgb = np.dstack((r,g,b)) #Stack to form RGB (with irg)
            rgb -= np.min(rgb)
            rgb = rgb / np.max(rgb)  #Normalise linearly to 0.0-1.0
            rgb = rgb * 255          #Scale to be 0-255 for saving
            img = Image.fromarray(rgb.astype(np.uint8), mode='RGB') #Make the simulation a PIL image
            
            if plot:
                print(np.max(np.dstack((r, g, b))))
                print(np.max(r), np.max(g), np.max(b))
                plt.imshow(img)
                plt.show()
                #plt.imshow(IMG)
                #plt.show()
                print(path_psf+objs[div][i][len(path):-len(extn)]+p+'.jpg')
                print(path_psf, objs[div][i][len(path)+len(div)+1:-len(extn)]+p)
                break
            
            #Save
            img.save(path_psf+objs[div][i][len(path):-len(extn)]+p+'.jpg')
        if plot:
            break
    if plot:
        break

    #Save galaxy name + projection and redshift used when reprojected
    cols = []
    cols.append(fits.Column(name='Object', format='30A', array=obj_name))
    cols.append(fits.Column(name='redshift', format='E', array=obj_zsft))
    tbhdu = fits.BinTableHDU.from_columns(cols)
    prihdr = fits.Header()
    prihdr['TITLE'] = div+'_cat'
    prihdr['CREATOR'] = 'WJP'
    prihdu = fits.PrimaryHDU(header=prihdr)
    fits.HDUList([prihdu, tbhdu]).writeto(path_psf+div+'_catalogue.fits', overwrite=True)
    print('Written catalogue to', path_psf+div+'_catalogue.fits')

