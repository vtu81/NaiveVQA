"""
Clean up questions and annotations without images!
"""
import os

dataDir		='.'
train_test_val = 'train' # only the train set need to be cleaned
dataSubType = 'train2014' if train_test_val == 'train' else 'val2014'
annFile     = './annotations/%s.json' %(train_test_val) # input
quesFile    = './questions/%s.json' %(train_test_val) # input
cleanAnnFile     = './annotations/%s_cleaned.json' %(train_test_val) # output
cleanQuesFile    = './questions/%s_cleaned.json' %(train_test_val) # output
imgDir 		= '%s/images/%s/' %(dataDir, train_test_val)

import json

with open(annFile,'r') as load_f:
    load_dict=json.load(load_f)
print('Original annotations num:', len(load_dict['annotations']))
clean = filter(lambda x: os.path.isfile(imgDir + 'COCO_' + dataSubType + '_'+ str(x['image_id']).zfill(12) + '.jpg'), load_dict["annotations"])
load_dict["annotations"] = list(clean)
print('Cleaned annotations num:', len(load_dict['annotations']))
with open(cleanAnnFile, 'w') as fp:
    fp.write(json.dumps(load_dict))


with open(quesFile, 'r') as load_f:
    load_dict=json.load(load_f)
print('Original questions num:', len(load_dict['questions']))
clean = filter(lambda x: os.path.isfile(imgDir + 'COCO_' + dataSubType + '_'+ str(x['image_id']).zfill(12) + '.jpg'), load_dict["questions"])
load_dict["questions"] = list(clean)
print('Cleaned questions num:', len(load_dict['questions']))
with open(cleanQuesFile, 'w') as fp:
    fp.write(json.dumps(load_dict))
