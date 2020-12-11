import tensorflow as tf
from collections import namedtuple
import numpy as np
import setGPU
from matplotlib import pyplot as plt
import vae.vae_particle as vap
import vae.losses as losses
import training as tra


def get_mnist_train_and_valid_dataset(batch_size=64):
    # Prepare the training dataset.
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    # Reserve 10,000 samples for validation.
    x_train = x_train.astype('float32') / 255.
    x_val = x_train[-3000:, 12:24, 0:25]
    x_train = x_train[:10000, 12:24, 0:25]
    x_train = np.reshape(x_train, (len(x_train),-1, 3))
    x_val = np.reshape(x_val, (len(x_val), -1, 3))

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
    val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)
    return train_dataset, val_dataset, np.mean(x_train), np.std(x_train)



input_shape = (100, 3)
Parameters = namedtuple('Parameters', 'beta train_total_n valid_total_n batch_n')
params = Parameters(beta=0.0001, train_total_n=int(10e5), valid_total_n=int(1e5), batch_n=64) # 'L1L2'

#### get data ####
train_ds, valid_ds, *mean_stdev = get_mnist_train_and_valid_dataset(params.batch_n)

#### training setup ####
# optimizer
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Instantiate a loss function.
loss_fn = losses.threeD_loss


#### build model ####

vae = vap.VAEparticle(input_shape=input_shape, z_sz=10, filter_ini_n=6, kernel_sz=3)
model = vae.build(mean_stdev)

#### train
model = tra.train(model=model, loss_fn=loss_fn, train_ds=train_ds, valid_ds=valid_ds, epochs=300, optimizer=optimizer, beta=params.beta, patience=5, min_delta=0.001)

#### show results
for i in range(3):
    img = next(valid_ds.as_numpy_iterator())
    plt.imshow(np.squeeze(img[0]), cmap='gray')
    plt.savefig('fig/test/orig'+str(i)+'.png')
    plt.clf(); plt.cla(); plt.close()
    img_pred = model.predict(img[0][np.newaxis,:,:])
    plt.imshow(np.squeeze(img_pred), cmap='gray')
    plt.savefig('fig/test/reco'+str(i)+'.png')
    plt.clf(); plt.cla(); plt.close()



