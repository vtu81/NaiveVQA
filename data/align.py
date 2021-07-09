"""
Align the annotation json files and corresponding question json files,
so that their `question_id` would match!
"""
import json

def align(train_test_val='train'):
    assert train_test_val == 'train' or train_test_val == 'val' or train_test_val == 'test'
    dataDir		='.'
    dataSubType = 'train2014' if train_test_val == 'train' else 'val2014'
    annFile     = './annotations/%s.json' %(train_test_val) # input
    quesFile    = './questions/%s.json' %(train_test_val) # input
    cleanAnnFile     = './annotations/%s_cleaned.json' %(train_test_val) # output
    cleanQuesFile    = './questions/%s_cleaned.json' %(train_test_val) # output
    imgDir 		= '%s/images/%s/' %(dataDir, train_test_val)

    print("")
    
    # Sort annotations
    with open(annFile,'r') as load_f:
        load_dict=json.load(load_f)
    print('Annotations num:', len(load_dict['annotations']))
    sorted_a = sorted(load_dict["annotations"], key=lambda x: (x['question_id']))
    load_dict["annotations"] = sorted_a
    with open(cleanAnnFile, 'w') as fp:
        fp.write(json.dumps(load_dict))

    # Sort questions
    with open(quesFile, 'r') as load_f:
        load_dict=json.load(load_f)
    print('Questions num:', len(load_dict['questions']))
    sorted_q = sorted(load_dict['questions'], key=lambda x: (x['question_id']))
    load_dict["questions"] = sorted_q
    with open(cleanQuesFile, 'w') as fp:
        fp.write(json.dumps(load_dict))


    # Validate
    qa_pairs = list(zip(sorted_q, sorted_a))
    not_matching_num = 0
    for q, a in qa_pairs:
        if q['question_id'] != a['question_id']:
            # print("Not matching: ", q['question_id'], a['question_id'])
            not_matching_num += 1

    if not_matching_num == 0:
        print("Successfully aligned \'{}\' & \'{}\' \n-> \'{}\' & \'{}\'".format(annFile, quesFile, cleanAnnFile, cleanQuesFile))
    else:
        print("{} and {} cannot be matched! Please check their correspondance.".format(annFile, quesFile))


if __name__ == '__main__':
    align('train')
    align('val')
    align('test')