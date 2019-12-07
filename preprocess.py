import os
import numpy
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from numpy import savez_compressed, vstack, asarray, load
import numpy as np
from matplotlib import pyplot
# make the dataset

def load_images(path, size=(256, 256)):
	data_list = []
	file_list = os.listdir(path)
	# enumerate filenames in directory, assume all are images
	i = 0
	for filename in file_list:
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# store
		#pixels2 = (pixels/255)
		#pyplot.imshow(pixels2)
		#pyplot.show()
		#pyplot.close()
		data_list.append(pixels)
		#pyplot.imshow(data_list[i]/255)
		#pyplot.show()
		#pyplot.close()
		i+=1
		print(i)
	data_list = asarray(data_list)
	# data_list = (data_list - 127.5)/127.5 #(data_list/127.5) - 1
	# data_list = data_list.astype(np.float16)
	return data_list

def load_dataset():
	data_male = load_images('male/')
	data_female = load_images('female/')
	# split to train and test datasets
	#train_male, train_female, test_male, test_female = train_test_split(data_male, data_female, test_size=0.25)
	#return train_male, train_female, test_male, test_female
	return data_male, data_female

def load_saved_dataset(name):
	data = load(name)
	x, y = data['arr_0'], data['arr_1']
	# scale to [-1, 1]
	# data = (data/127.5) - 1
	# x = (x/127.5) - 1
	# y = (y/127.5) - 1
	return x, y
	
def main():
	print('Eating dataset')
	#a, b, A, B = load_dataset()
	a, b = load_dataset()
	pyplot.imshow(a[0]/255)
	pyplot.show()
	pyplot.imshow(a[1]/255)
	pyplot.show()
	pyplot.imshow(a[2]/255)
	pyplot.show()
	pyplot.imshow(a[3]/255)
	pyplot.show()
	pyplot.imshow(b[0]/255)
	pyplot.show()
	pyplot.imshow(b[1]/255)
	pyplot.show()
	pyplot.imshow(b[2]/255)
	pyplot.show()
	pyplot.imshow(b[3]/255)
	pyplot.show()
	savez_compressed('proj_dataset.npz', a, b)
	print('Finished')
	'''
	# visualize loaded dataset
	a, b = load_saved_dataset('proj_dataset.npz')
	imgs = np.concatenate(a[:8], b[:8])
	imgs = imgs.reshape((4, 4, 256, 256, 3))
	imgs = np.vstack([np.hstack(i) for i in imgs])
	plt.figure()
	plt.axis('off')
	plt.title('Input: 1st 2 rows, Decoded: last 2 rows')
	plt.imshow(imgs, interpolation='none', cmap='rgb')
	plt.savefig('input_and_decoded.png')
	plt.show()
	'''

if __name__ == "__main__":
	main()