import tensorflow as tf
from tensorflow.contrib import predictor

from astropy.io import fits
from PIL import Image
import numpy as np
import pickle

import glob


#Set data
sourc = 'sdss'    #'eagle' 'sdss'
if sourc == 'eagle':
    path = '/Volumes/Data_Disk/sky_maps/sdss/eagle-nodust-redshifted/'
    division = ['merger_0.1', 'nomerger']
    extn = '.jpg'
    outfile = './custom-network-eagle-time/'+division[0]
    edge_cut = 0
elif sourc == 'sdss':
    path = '/Volumes/Data_Disk/sky_maps/sdss/cutouts/'
    division = ['mergers', 'non-mergers']
    extn = '.jpeg'
    outfile = './custom-network-sdss/'+division[0]
    edge_cut = 96
    
no_dim = len(division)

epochs = 100
batch_size = 64
wiggle = False

#Validation or test
go_type = 'test' #'validation' 'test'


#Get objects
objs = {}
length = 1e8
for div in division:
    objs[div] = sorted(glob.glob(path+div+'/*'+extn))
    np.random.seed(0)
    np.random.shuffle(objs[div])
    print(path+div+'/*'+extn)
    if len(objs[div]) < length:
        length = len(objs[div])
tenpct = length // 10
no_div = len(division)


#function to get next batch
def next_batch(objs, division, batch_size=64, typ='train', wiggle=False):
    clas = len(objs)
    
    if typ == 'test':
        lower = 0
        upper = tenpct
    elif typ == 'validation':
        lower = tenpct
        upper = 2*tenpct
    else:
        lower = 2*tenpct
        upper = length
        
    t_indx = np.arange(lower, upper)
    
    np.random.shuffle(t_indx)
    
    if len(t_indx) < batch_size//clas:
        batch_size = int(clas * len(t_indx))
    
    x_train = []
    y_train = []
    n_train = []
    for h in range(0, len(division)):
        div = division[h]
        for i in range(0, batch_size//clas):
            n_train.append(int(objs[div][t_indx[i]][len(path+div)+2:-len(extn)]))
            img = Image.open(objs[div][t_indx[i]])
            if edge_cut == 0:
                x_train.append(np.array(img)/255.)
            else:
                x_train.append(np.array(img)[edge_cut:-edge_cut,edge_cut:-edge_cut]/255.)
            if wiggle:
                np.rot90(x_train[-1], np.random.randint(4, size=1))
                if np.random.randint(1):
                    np.fliplr(x_train[-1])
                if np.random.randint(1):
                    np.flipud(x_train[-1])
            y_train.append(h)
            
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    n_train = np.array(n_train)
    
    batch_size = len(x_train)
    t_indx = np.arange(0, batch_size)
    np.random.shuffle(t_indx)
    
    x_train = x_train[t_indx]
    y_train = y_train[t_indx]
    n_train = n_train[t_indx]
    
    return x_train, y_train, len(t_indx), n_train


#Load saved model
my_model = predictor.from_saved_model(outfile)


#Generate batch and assess them
x_batch, y_batch, num, n_batch = next_batch(objs, division, batch_size=2*tenpct, wiggle=False, typ=go_type)
pred = my_model({"X":x_batch, "is_training":False})
pred = np.argmax(pred['y_prob'], axis=1)


#Count correct and incorrectly identified objects at cut=0.5
corr = 0
for j in range(0, no_div):
    testing = np.where(pred == j)
    correct = np.where(np.logical_and(pred == j, y_batch == j))
    validating = np.where(y_batch == j)
    
    corr += len(correct[0])

    if len(testing[0]) == 0:
        cor_tes = 'inf'
    elif len(correct[0]) == 0:
        cor_tes = 0.0
    else:
        cor_tes = len(correct[0])/len(testing[0])
        cor_tes = round(cor_tes, 3)

    if len(validating[0]) == 0:
        cor_val = 'inf'
    elif len(correct[0]) == 0:
        cor_val = 0.0
    else:
        cor_val = len(correct[0])/len(validating[0])
        cor_val = round(cor_val, 3)

    if type(cor_val) == 'float':
        print("Val \t Are {0} classed {0} (Recall): {1:.3g} ({2:.3g} of {3:.3g})".format(division[j], cor_val, len(correct[0]), len(validating[0])))
    else:
        print("Val \t Are {0} classed {0} (Recall): {1} ({2:.3g} of {3:.3g})".format(division[j], cor_val, len(correct[0]), len(validating[0])))
    if type(cor_tes) == 'float':
        print("Val \t Classed {0} are {0} (Precision): {1:.3g} ({2:.3g} of {3:.3g})".format(division[j], cor_tes, len(correct[0]), len(testing[0])))
    else:
        print("Val \t Classed {0} are {0} (Precision): {1} ({2:.3g} of {3:.3g})".format(division[j], cor_tes, len(correct[0]), len(testing[0])))
          
loss = 9.999
print("Val \t Overall loss = {0:.3g} and Accuracy of {1:.3g} ({2} of {3})".format(loss, corr/num, corr, num))


#Reload batch and rerun classification
x_batch, y_batch, num, n_batch= next_batch(objs, division, batch_size=2*tenpct, wiggle=False, typ=go_type)
pred = my_model({"X":x_batch, "is_training":False})
pred = pred['y_prob']


#Save results
cols = []
cols.append(fits.Column(name='Object', format='K', array=n_batch))
cols.append(fits.Column(name='True_Class', format='E', array=y_batch))
cols.append(fits.Column(name='merger_frac', format='E', array=pred[:,0]))
tbhdu = fits.BinTableHDU.from_columns(cols)
prihdr = fits.Header()
prihdr['TITLE'] = go_type+'_cat'
prihdr['CREATOR'] = 'WJP'
prihdu = fits.PrimaryHDU(header=prihdr)
fits.HDUList([prihdu, tbhdu]).writeto('./cats_for_plots/'+sourc+'_'+go_type+'_catalogue.fits', overwrite=True)
print('Written catalogue to ./cats_for_plots/'+sourc+'_'+go_type+'_catalogue.fits')

print(pred.shape)

#Use different cuts to generate ROC
step = 0.01
cuts = np.arange(0.0, 1.0+step, step)
mgr = np.where(y_batch == 0)[0]
nmg = np.where(y_batch == 1)[0]

tpr = []
fpr = []
tnr = []
fnr = []
ppv = []
npv = []
end = 0

for cut in cuts:
    cmg = np.where(pred[:,0] > cut)[0] #classed merger
    if len(cmg) == 0:
        print('cmg', cut)
        end += 1
        continue
    cnm = np.where(pred[:,0] <= cut)[0] #classed non merger
    if len(cmg) == 0:
        print('cnm', cut)
        end += 1
        continue
    
    #Positive
    tp = np.intersect1d(cmg, mgr) #classed merger are merger
    fp = np.intersect1d(cmg, nmg) #classed merger are non merger
    #Negative
    tn = np.intersect1d(cnm, nmg) #classed non merger are non merger
    fn = np.intersect1d(cnm, mgr) #classed non merger are merger
    
    tpr.append(len(tp)/(len(tp)+len(fn)))
    fpr.append(len(fp)/(len(fp)+len(tn)))
    
    tnr.append(len(tn)/(len(tn)+len(fp)))
    fnr.append(len(fn)/(len(fn)+len(tp)))
    
    ppv.append(len(tp)/(len(tp)+len(fp)))
    #npv.append(len(tn)/(len(tn)+len(fn)))
    
tpr = np.array(tpr)
fpr = np.array(fpr)
tnr = np.array(tnr)
fnr = np.array(fnr)
ppv = np.array(ppv)


#Plot ROC
import matplotlib.pyplot as plt

plt.plot(fpr, tpr, 'o', label='positive')
plt.plot([0.0, 1.0],[0.0, 1.0], '--')
plt.title("ROC: FPR vs TPR")
plt.xlabel('False Positive Rate (Fall Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.show()

order = np.argsort(fpr)
fpr_sort = np.array(fpr)[order]
tpr_sort = np.array(tpr)[order]
auc = np.trapz(tpr_sort, fpr_sort)
print('Area under ROC:', round(auc, 5))
roc = np.sqrt(np.square(fpr)+np.square(1-tpr))
bst = cuts[np.argmin(roc)]
print('Smallest ROC of', round(np.min(roc), 5), 'at', bst, '(', np.argmin(roc), ')')
if go_type == 'test' and sourc == 'sdss':
    bst = 0.57

#Save ROC data
with open('./pickle_for_plots/ROC_'+sourc+'_'+go_type+'.pkl', "wb") as f:
    pickle.dump({'fpr':fpr_sort, 'tpr':tpr_sort}, f)
    f.close()

#Plot a few other plots
plt.plot(tpr, ppv, 'o', label='positive')
#plt.plot([0.0, 1.0],[0.0, 1.0], '--')
plt.title("PPV vs TPR")
plt.ylabel('Positive Predictive Value (Precision)')
plt.xlabel('True Positive Rate (Recall)')
#plt.legend(loc=4, frameon=False)
plt.show()

plt.plot(cuts[:-end], tpr, 'o', label='TPR')
plt.axhline(0.8)
plt.plot(cuts[:-end], fpr, 'o', label='FPR')
plt.axhline(0.2)
plt.axvline(bst)
plt.xlabel('Cut')
plt.ylabel('Rate')
plt.legend(loc=0, frameon=False)
plt.show()


#Print statistics at each cut point
are_classed = np.empty((len(cuts), no_div))
classed_are = np.empty((len(cuts), no_div))
overall = np.empty(len(cuts))

for i in range(0, len(cuts)):
    print(cuts[i])
    new_pred = np.zeros(len(pred))
    non_merger = np.where(pred[:,0] <= cuts[i])[0]
    new_pred[non_merger] = 1

    corr = 0
    for j in range(0, no_div):
        testing = np.where(new_pred == j)
        correct = np.where(np.logical_and(new_pred == j, y_batch == j))
        validating = np.where(y_batch == j)

        corr += len(correct[0])

        if len(testing[0]) == 0:
            cor_tes = 'inf'
        elif len(correct[0]) == 0:
            cor_tes = 0.0
        else:
            cor_tes = len(correct[0])/len(testing[0])
            cor_tes = round(cor_tes, 3)

        if len(validating[0]) == 0:
            cor_val = 'inf'
        elif len(correct[0]) == 0:
            cor_val = 0.0
        else:
            cor_val = len(correct[0])/len(validating[0])
            cor_val = round(cor_val, 3)

        if type(cor_val) == 'float':
            print("Val \t Are {0} classed {0} (Recall): {1:.3g} ({2:.3g} of {3:.3g})".format(division[j], cor_val, len(correct[0]), len(validating[0])))
        else:
            print("Val \t Are {0} classed {0} (Recall): {1} ({2:.3g} of {3:.3g})".format(division[j], cor_val, len(correct[0]), len(validating[0])))
        if type(cor_tes) == 'float':
            print("Val \t Classed {0} are {0} (Precision): {1:.3g} ({2:.3g} of {3:.3g})".format(division[j], cor_tes, len(correct[0]), len(testing[0])))
        else:
            print("Val \t Classed {0} are {0} (Precision): {1} ({2:.3g} of {3:.3g})".format(division[j], cor_tes, len(correct[0]), len(testing[0])))
        are_classed[i,j] = cor_val
        classed_are[i,j] = cor_tes

    print("Val \t Overall loss = {0:.3g} and Accuracy of {1:.3g} ({2} of {3})".format(loss, corr/num, corr, num))
    overall[i] = corr/num
    print()


#Another nice plot
plt.plot(cuts, are_classed[:,0], label='Are mgr classed mgr')
plt.plot(cuts, are_classed[:,1], label='Are nmg classed nmg')
plt.plot(cuts, classed_are[:,0], label='Classed mgr are mgr')
plt.plot(cuts, classed_are[:,1], label='Classed nmg are nmg')
plt.plot(cuts, overall, label='Overall')
plt.xlabel('Cut')
plt.legend(frameon=False)
plt.axvline(bst)
#plt.savefig("sdss-val_stats.png", bbox_inches="tight")
plt.show()


#Print statistics of best cut
print(bst)
cut_idx = np.where(cuts == bst+0.01)[0]-1
#cut_idx = np.where(cuts == bst)[0]
print(cut_idx)
print('TPR:', tpr[cut_idx[0]])
print('FPR:', fpr[cut_idx[0]])
print('Are merger classed merger (Recall):', are_classed[cut_idx,0])
print('Classed merger are merger (Precision):', classed_are[cut_idx,0])
print()
print(cut_idx)
print('TNR:', tnr[cut_idx[0]])
print('FNR:', fnr[cut_idx[0]])
print('Are non-merger classed non-merger (Recall):', are_classed[cut_idx,1])
print('Classed non-merger are non-merger (Precision):', classed_are[cut_idx,1])
print()
print("Accuracy:", overall[cut_idx])

