### INITIALIZATION ###
import numpy
from numpy import load
from random import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from matplotlib import pyplot
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam

EPOCHS = 50
POOL_SIZE = 50

# we'll need this
from models import make_discriminator_model, make_generator_model, define_composite_model
from preprocess import load_saved_dataset

### SUPPORT FUNCTIONS ###

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return X, y
 
# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, dataset, patch_shape):
	# generate fake instance
	X = g_model.predict(dataset)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y
 
# save the generator models to file
def save_models(step, g_model_AtoB, g_model_BtoA):
	# save the first generator model
	filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
	g_model_AtoB.save_weights(filename1)
	# save the second generator model
	filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
	g_model_BtoA.save_weights(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))
 
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, trainX, name, n_samples=5):
	# select a sample of input images
	X_in, _ = generate_real_samples(trainX, n_samples, 0)
	# generate translated images
	X_out, _ = generate_fake_samples(g_model, X_in, 0)
	# scale all pixels from [-1,1] to [0,1]
	X_in = (X_in + 1) / 2.0
	X_out = (X_out + 1) / 2.0
	# plot real images
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_in[i])
	# plot translated image
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_out[i])
	# save plot to file
	filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
	pyplot.savefig(filename1)
	pyplot.close()
 
# update image pool for fake images
def update_image_pool(pool, images, max_size=POOL_SIZE):
	selected = list()
	for image in images:
		if len(pool) < max_size:
			# stock the pool
			pool.append(image)
			selected.append(image)
		elif random() < 0.5:
			# use image, but don't add it to the pool
			selected.append(image)
		else:
			# replace an existing image and use replaced image
			ix = randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return asarray(selected)
 
# train cyclegan models
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
	# loss vars
	g1_loss = numpy.array([], numpy.float16)
	g2_loss = numpy.array([], numpy.float16)
	d1A_loss = numpy.array([], numpy.float16)
	d2A_loss = numpy.array([], numpy.float16)
	d1B_loss = numpy.array([], numpy.float16)
	d2B_loss = numpy.array([], numpy.float16)
	# define properties of the training run
	n_epochs, n_batch, = EPOCHS, 1
	# determine the output square shape of the discriminator
	n_patch = d_model_A.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# prepare image pool for fakes
	poolA, poolB = list(), list()
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
		X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
		X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
		# update fakes from pool
		X_fakeA = update_image_pool(poolA, X_fakeA)
		X_fakeB = update_image_pool(poolB, X_fakeB)
		# update generator B->A via adversarial and cycle loss
		g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
		# update discriminator for A -> [real/fake]
		dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
		dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
		# update generator A->B via adversarial and cycle loss
		g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
		# update discriminator for B -> [real/fake]
		dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
		dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
		# summarize performance
		print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
		d1A_loss = numpy.append(d1A_loss, dA_loss1)
		d2A_loss = numpy.append(d2A_loss, dA_loss2)
		d1B_loss = numpy.append(d1B_loss, dB_loss1)
		d2B_loss = numpy.append(d2B_loss, dB_loss2)
		g1_loss = numpy.append(g1_loss, g_loss1)
		g2_loss = numpy.append(g2_loss, g_loss2)
		# evaluate the model performance every so often
		if (i+1) % (bat_per_epo * 1) == 0:
			# plot A->B translation
			# summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
			# plot B->A translation
			# summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
			save_models(i, g_model_AtoB, g_model_BtoA)	# save model every epoch instead of predicting
			# dump losses
			numpy.savetxt("disc_male_real.csv", d1A_loss, delimiter=",")
			numpy.savetxt("disc_male_fake.csv", d2A_loss, delimiter=",")
			numpy.savetxt("disc_female_real.csv", d1B_loss, delimiter=",")
			numpy.savetxt("disc_female_fake.csv", d2B_loss, delimiter=",")
			numpy.savetxt("gen_m2f.csv", g1_loss, delimiter=",")
			numpy.savetxt("gen_f2m.csv", g2_loss, delimiter=",")
		'''
		if (i+1) % (bat_per_epo * 5) == 0:
			# save the models
			save_models(i, g_model_AtoB, g_model_BtoA)
		'''

### LOAD DATASET ### 
 
# load image data
m, f = load_saved_dataset('proj_dataset.npz')
dataset = [m, f]
print('Loaded ', m.shape, f.shape)
# define input shape based on the loaded dataset
image_shape = m.shape[1:]

### MAKE GENERATOR ###

# generator: A -> B
g_model_AtoB = make_generator_model(6)
# generator: B -> A
g_model_BtoA = make_generator_model(6)

# print model
# plot_model(g_model_AtoB, to_file='generator.png', show_shapes=True, show_layer_names=True)

### MAKE DISCRIMINATOR ###

# discriminator: A -> [real/fake]
d_model_A = make_discriminator_model()
# discriminator: B -> [real/fake]
d_model_B = make_discriminator_model()

# print model
# plot_model(d_model_A, to_file='discriminator.png', show_shapes=True, show_layer_names=True)

### PUT THE MODELS TOGETHER TO COMPLETE THE GAN ###

# composite: A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)

# second compile to fix a weird error
c_model_AtoB.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=Adam(lr=0.0002, beta_1=0.5))
c_model_BtoA.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=Adam(lr=0.0002, beta_1=0.5))

# print model
# plot_model(c_model_AtoB, to_file='composite.png', show_shapes=True, show_layer_names=True)

### TRAIN MODEL ###

# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)

### SHOW EXAMPLES SUPPORT FUNCTIONS ###

### showing examples ###
# load and prepare training images
'''
def load_real_samples(filename):
	# load the dataset
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]
''' 
# select a random sample of images from the dataset
def select_sample(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	return X
 
# plot the image, the translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
	images = vstack((imagesX, imagesY1, imagesY2))
	titles = ['Real', 'Generated', 'Reconstructed']
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, len(images), 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# title
		pyplot.title(titles[i])
	pyplot.show()

### SHOW EXAMPLE(S) ###	
	
# pushed off to notebook	

'''	
# load dataset
A_data, B_data = load_real_samples('horse2zebra_256.npz')
print('Loaded', A_data.shape, B_data.shape)
# load the models
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('g_model_AtoB_089025.h5', cust)
model_BtoA = load_model('g_model_BtoA_089025.h5', cust)

# plot A->B->A
A_real = select_sample(A_data, 1) # change 1 to n. images desired
B_generated  = g_AtoB.predict(A_real)
A_reconstructed = g_BtoA.predict(B_generated)
show_plot(A_real, B_generated, A_reconstructed)
# plot B->A->B
B_real = select_sample(B_data, 1)
A_generated  = g_BtoA.predict(B_real)
B_reconstructed = g_AtoB.predict(A_generated)
show_plot(B_real, A_generated, B_reconstructed)

'''