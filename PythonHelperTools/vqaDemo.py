from vqaTools.vqa import VQA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os

dataDir		='../data'
train_test_val = 'train'
dataSubType = 'train2014' if train_test_val == 'train' else 'val2014'
annFile     = '%s/annotations/%s.json'%(dataDir, train_test_val)
quesFile    = '%s/questions/%s.json'%(dataDir, train_test_val)
imgDir 		= '%s/images/%s/' %(dataDir, train_test_val)

# initialize VQA api for QA annotations
vqa=VQA(annFile, quesFile)

# load and display QA annotations for given question types
"""
All possible quesTypes for abstract and mscoco has been provided in respective text files in ../QuestionTypes/ folder.
"""
annIds = vqa.getQuesIds(quesTypes='how many');   
anns = vqa.loadQA(annIds)
randomAnn = random.choice(anns)
vqa.showQA([randomAnn])
imgId = randomAnn['image_id']
imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
if os.path.isfile(imgDir + imgFilename):
	I = io.imread(imgDir + imgFilename)
	plt.imshow(I)
	plt.axis('off')
	plt.show()
else:
	print('File ', imgFilename, ' not exists!')

# load and display QA annotations for given answer types
"""
ansTypes can be one of the following
yes/no
number
other
"""
annIds = vqa.getQuesIds(ansTypes='yes/no');   
anns = vqa.loadQA(annIds)
randomAnn = random.choice(anns)
vqa.showQA([randomAnn])
imgId = randomAnn['image_id']
imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
if os.path.isfile(imgDir + imgFilename):
	I = io.imread(imgDir + imgFilename)
	plt.imshow(I)
	plt.axis('off')
	plt.show()

# load and display QA annotations for given images
"""
Usage: vqa.getImgIds(quesIds=[], quesTypes=[], ansTypes=[])
Above method can be used to retrieve imageIds for given question Ids or given question types or given answer types.
"""
ids = vqa.getImgIds()
annIds = vqa.getQuesIds(imgIds=random.sample(ids,5));  
anns = vqa.loadQA(annIds)
randomAnn = random.choice(anns)
vqa.showQA([randomAnn])  
imgId = randomAnn['image_id']
imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
if os.path.isfile(imgDir + imgFilename):
	I = io.imread(imgDir + imgFilename)
	plt.imshow(I)
	plt.axis('off')
	plt.show()

# Check dataset completeness
"""
Uncleaned dataset:
	train:
		Existing images:  23071
		Not existing images:  21304
	test:
		Existing images:  21435
		Not existing images:  0
	val:
		Existing images:  21435
		Not existing images:  0

Cleaned dataset:
	train:
		Existing images:  23071
		Not existing images:  0
	test:
		Existing images:  21435
		Not existing images:  0
	val:
		Existing images:  21435
		Not existing images:  0
"""
annIds = vqa.getQuesIds()
anns = vqa.loadQA(annIds)
exist_num = 0
not_exist_num = 0
for ann in anns:	
	imgId = ann['image_id']
	imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
	if os.path.isfile(imgDir + imgFilename):
		exist_num += 1
	else:
		not_exist_num += 1
print('Existing images: ', exist_num)
print('Not existing images: ', not_exist_num)