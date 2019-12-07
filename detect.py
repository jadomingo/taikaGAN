import cv2
import sys
import os
import os.path
import shutil
from PIL import Image
# from resizeimage import resizeimage

def detect(filename, outname, cascade_file = "./lbpcascade_animeface.xml"):
	if not os.path.isfile(cascade_file):
		raise RuntimeError("%s: not found" % cascade_file)

	cascade = cv2.CascadeClassifier(cascade_file)
	image = cv2.imread(filename, cv2.IMREAD_COLOR)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.equalizeHist(gray)

	faces = cascade.detectMultiScale(gray,
									 # detector options
									 scaleFactor = 1.1,
									 minNeighbors = 5,
									 minSize = (24, 24))
	if len(faces) > 0:
		x, y, w, h = faces[0]
		#print(x, y, w, h)
		try:
			cv2.imwrite(outname, image[int(y-0.35*h): int(y+1.15*h), int(x-0.25*w): int(x+1.25*w)])
		except:
			return False
		return True
	else:
		return False

ct = 0
try:
	os.mkdir('crop_male')
except:
	pass
try:
	os.mkdir('crop_female')
except:
	pass

'''
for y in range(2015, 2020):
	img_dir = './images/' + str(y)
	files = os.listdir(./male/)
	for f in files:
		if detect(os.path.join(img_dir, f), './cropped/{}_{}.jpg'.format(ct, y)):
			with Image.open('./cropped/{}_{}.jpg'.format(ct, y), 'r') as r_img:
				# resize to 128px by 128px
				t_img = resizeimage.resize_contain(r_img, [128,128])
				t_img = t_img.convert('RGB') # get rid of transparency layer
				t_img.save('./crop_2/{}_{}.jpg'.format(ct, y), r_img.format)
				r_img.close()
			ct += 1
			print(ct)
'''
			
male_files = os.listdir('./male/')
female_files = os.listdir('./female/')
for f in male_files:
	if detect('./male/{}'.format(f), './crop_male/{}'.format(f)):
		with Image.open('./crop_male/' + f) as r_img:
			# resize to 128px by 128px
			t_img = r_img.resize([128,128])
			t_img = t_img.convert('RGB') # get rid of transparency layer
			t_img.save('./crop_male/{}'.format(f), r_img.format)
			r_img.close()
			ct += 1
print('male: ' + str(ct))
for f in female_files:
	if detect('./female/{}'.format(f), './crop_female/{}'.format(f)):
		with Image.open('./crop_female/' + f) as r_img:
			# resize to 128px by 128px
			t_img = r_img.resize([128,128])
			t_img = t_img.convert('RGB') # get rid of transparency layer
			t_img.save('./crop_female/{}'.format(f), r_img.format)
			r_img.close()
			ct += 1
print('female: ' + str(ct))