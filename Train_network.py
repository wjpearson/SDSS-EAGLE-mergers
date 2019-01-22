import tensorflow as tf
import numpy as np

import glob
import os

from PIL import Image

#Set source information
sourc = 'eagle'    #'eagle' 'sdss'
if sourc == 'eagle':
    path = './eagle-final/eagle-nodust2-zrcn/'
    division = ['merger_0.1', 'nomerger'] #'premerger', 'postmerger', 
    extn = '.jpg'
    outfile = './custom-network-eagle-zrcn/'+division[0]
    edge_cut = 0
elif sourc == 'sdss':
    path = './images-sdss/'
    division = ['mergers', 'non-mergers'] 
    extn = '.jpeg'
    outfile = './custom-network-sdss/'+division[0]
    edge_cut = 96

no_dim = len(division)
save = True
overwrite_existing = True

epochs = 200
batch_size = 64
wiggle = True

#find smallest group and match size to that
objs = {}
length = 1e8
for div in division:
    objs[div] = sorted(glob.glob(path+div+'/*'+extn))
    np.random.seed(0)
    np.random.shuffle(objs[div])
    if len(objs[div]) < length:
        length = len(objs[div])
tenpct = length // 10

epoch_length = (length - (2*tenpct) ) // batch_size
no_div = len(division)

print(length)


# In[4]:


#get image properties
img = Image.open(objs[division[0]][0])
img = np.array(img)
px = img.shape[0] - (2*edge_cut)
ch = img.shape[2]
print('px:', px, 'ch:', ch)


#Function to generate next batch
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
    for h in range(0, len(division)):
        div = division[h]
        for i in range(0, batch_size//clas):
            img = Image.open(objs[div][t_indx[i]])
            img = np.array(img)
            if edge_cut != 0:
                img = img[edge_cut:-edge_cut,edge_cut:-edge_cut]
            img = img/np.max(img)
            x_train.append(img)
            if wiggle:
                np.rot90(x_train[-1], np.random.randint(4, size=1))
                if np.random.randint(1):
                    np.fliplr(x_train[-1])
                if np.random.randint(1):
                    np.flipud(x_train[-1])
            y_train.append(h)
            
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    batch_size = len(x_train)
    t_indx = np.arange(0, batch_size)
    np.random.shuffle(t_indx)
    
    x_train = x_train[t_indx]
    y_train = y_train[t_indx]

    return x_train, y_train, len(t_indx)

#define CNN
def morph_model(x, y, training=False):
    drop_rate = 0.2
    print(x.shape)
    conv1 = tf.layers.conv2d(x, 32, 6, strides=1, padding='same', name='conv1')
    relu1 = tf.nn.relu(tf.layers.batch_normalization(conv1), name='relu1')
    drop1 = tf.layers.dropout(relu1, rate=drop_rate, training=training, name='drop1')
    print(drop1.shape)
    pool1 = tf.layers.max_pooling2d(drop1, 2, 2, padding='same', name='pool1')
    
    conv2 = tf.nn.relu(tf.layers.conv2d(pool1, 64, 5, strides=1, padding='same'), name='conv2')
    relu2 = tf.nn.relu(tf.layers.batch_normalization(conv2), name='relu2')
    drop2 = tf.layers.dropout(relu2, rate=drop_rate, training=training, name='drop2')
    print(drop2.shape)
    pool2 = tf.layers.max_pooling2d(drop2, 2, 2, padding='same', name='pool2')
    
    conv3 = tf.layers.conv2d(pool2, 128, 3, strides=1, padding='same', name='conv3')
    relu3 = tf.nn.relu(tf.layers.batch_normalization(conv3), name='relu3')
    drop3 = tf.layers.dropout(relu3, rate=drop_rate, training=training, name='drop3')
    print(drop3.shape)
    conv4 = tf.layers.conv2d(drop3, 128, 3, strides=1, padding='same', name='conv4')
    relu4 = tf.nn.relu(tf.layers.batch_normalization(conv4), name='relu4')
    drop4 = tf.layers.dropout(relu4, rate=drop_rate, training=training, name='drop4')
    pool3 = tf.layers.max_pooling2d(drop4, 2, 2, padding='same', name='pool3')
    print(drop4.shape)
    
    flatten = tf.reshape(pool3, (-1, pool3.shape[1]*pool3.shape[2]*pool3.shape[3]))
    fc1 = tf.layers.dense(flatten, 2048, name='fc1')
    relu5 = tf.nn.relu(tf.layers.batch_normalization(fc1), name='relu5')
    drop5 = tf.layers.dropout(relu5, rate=drop_rate, training=training, name='drop5')
    fc2 = tf.layers.dense(drop5, 2048, name='fc2')
    relu6 = tf.nn.relu(tf.layers.batch_normalization(fc2), name='relu6')
    drop6 = tf.layers.dropout(relu6, rate=drop_rate, training=training, name='drop6')
    y_out = tf.layers.dense(drop6, len(division), name='y_out')
    
    return y_out


tf.reset_default_graph()
#Create place holders
X = tf.placeholder(tf.float32, [None, px, px, ch])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

y_out = morph_model(X,y,training=is_training)
y_prob = tf.nn.softmax(y_out)
#Set losses
total_loss = tf.losses.softmax_cross_entropy(tf.one_hot(y,no_div),logits=y_out)
mean_loss = tf.reduce_mean(total_loss)

optimizer = tf.train.AdamOptimizer(5e-5) # select optimizer and set learning rate
train_step = optimizer.minimize(mean_loss)

#Start tf session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(y_out,1), y)

#data to deciding when to save the model
i = 0
acc_old = 0.6
loss_old = 2.0
max_acc = 0.0
avg_acc = 0.0
#Train model
for i in range(0, epochs):
    ttl_usd = 0
    ttl_los = 0
    ttl_cor = 0
    for j in range(0, epoch_length):
        x_batch, y_batch, num = next_batch(objs, division, batch_size=batch_size, wiggle=wiggle)
        _, loss, corr = sess.run([train_step, mean_loss, correct_prediction],
                                 feed_dict = {X: x_batch, y: y_batch, is_training: True})
        ttl_usd += num
        ttl_los += (loss*num)
        ttl_cor += sum(corr)
    
    print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"              .format(ttl_los/ttl_usd, ttl_cor/ttl_usd, i+1))
    
    x_val, y_val, num = next_batch(objs, division, batch_size=2*tenpct, typ='validation')
    y_out_val, loss, corr = sess.run([y_prob, mean_loss, correct_prediction],
                                     feed_dict = {X: x_val, y: y_val, is_training: False})
    y_out_val_agm = np.argmax(y_out_val, axis=1)
    
    print("Val \t Overall loss = {0:.3g} and accuracy of {1:.3g} ({2} of {3})"          .format(loss, sum(corr)/num, sum(corr), num))
    avg_acc += sum(corr)/num
    if sum(corr)/num > max_acc:
        max_acc = sum(corr)/num

    for j in range(0, no_div):
        testing = np.where(y_out_val_agm == j)
        correct = np.where(np.logical_and(y_out_val_agm == j, y_val == j))
        validating = np.where(y_val == j)

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
        if sum(corr)/num > 0.9:
            print('Val \t Are', division[j], ' classed ', division[j], ':', cor_val, 
                  '(',len(correct[0]), 'of', len(validating[0]),')')
            print('Val \t Classed', division[j], ' are ', division[j], ':', cor_tes, 
                  '(',len(correct[0]), 'of', len(testing[0]),')')

    #Save if accuracy has increased AND loss has decreased
    if save and i > 1 and sum(corr)/num >= acc_old and loss <= loss_old:
        acc_old = sum(corr)/num
        loss_old = loss
        if overwrite_existing and os.path.exists(outfile):
            import shutil
            shutil.rmtree(outfile)
        tf.saved_model.simple_save(sess, outfile, {"X":X, "y":y, "is_training":is_training}, {"y_prob":y_prob})
        print('Saved')
        
    print()

#Print average and maximum accuracy
print(round(avg_acc/epochs, 3), round(max_acc, 3))

