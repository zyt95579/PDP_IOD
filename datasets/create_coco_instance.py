#from coco_hug import create_task_json, task_info_coco
import argparse
from pathlib import Path
import json
import os
import numpy as np

def task_info_coco(split_point=0):  # 0, 40, 70

    T1_CLASS_NAMES = [
        "person",
        "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase",
        "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle"
    ]  # n_classes = 19

    T2_CLASS_NAMES = [
        "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed"
    ]  # n_classes = 21

    T3_CLASS_NAMES = [
        "dining table", "toilet",
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    ]  # n_classes = 20

    # T4_CLASS_NAMES = [
    #     "harbor","bridge","storage tank"
    # ] # n_classes = 20

    all_classes = T1_CLASS_NAMES + T2_CLASS_NAMES + T3_CLASS_NAMES

    if split_point == 0:
        task_map = {1: (T1_CLASS_NAMES, 0, 40), 2: (T2_CLASS_NAMES, 40, 20),
                    3: (T3_CLASS_NAMES, 60, 20)}
    else:
        # np.random.shuffle(all_classes)
        task_map = {1: (all_classes[0:split_point], 0, split_point),
                    2: (all_classes[split_point:], split_point, 80 - split_point)}

    task_label2name = {}
    for i, j in enumerate(all_classes):
        task_label2name[i] = j

    return task_map, task_label2name

def task_info_voc(split_point=10):
    """
    为 PASCAL VOC 定义增量学习任务，并将类别列表定义在函数内部。

    Args:
        split_point (int): 任务划分点，代表第一阶段基础类的数量。
                             常用的值有 19, 15, 10。

    Returns:
        tuple: 包含 task_map 和 task_label2name 两个字典。
    """
    # --- PASCAL VOC 数据集的所有类别，顺序固定 (已移入函数内部) ---
    _all_classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    # 根据 split_point 动态切分基础类和新类
    T1_CLASS_NAMES = _all_classes[:split_point]
    T2_CLASS_NAMES = _all_classes[split_point:]

    all_classes = T1_CLASS_NAMES + T2_CLASS_NAMES

    # 动态创建任务地图
    task_map = {
        1: (T1_CLASS_NAMES, 0, split_point),
        2: (T2_CLASS_NAMES, split_point, 20 - split_point),
    }

    # 创建从标签索引到类别名的映射
    task_label2name = {i: name for i, name in enumerate(all_classes)}

    return task_map, task_label2name

def create_task_json(root_json, cat_names, set_type='train', offset=0, task_id=1, output_dir='', task_label2name=None):

    print ('Creating temp JSON for tasks ',task_id,' ...', set_type)
    temp_json = json.load(open(root_json, 'r'))

    id2id, name2id, flag = {}, {}, False

    if task_label2name:
        flag = True
        for i,j in task_label2name.items():
            name2id[j] = i

    cat_ids, keep_imgs = [], []
    for k,j in enumerate(temp_json['categories']):

        if not flag:
            name2id[j['name']] = j['id']

        if j['name'] in cat_names:
            cat_ids.append(j['id'])
    
    for j,i in enumerate(cat_names):
        id2id[name2id[i]] = offset+j
    
    data = {'images':[], 'annotations':[], 'categories':[], 
            'info':{},'licenses':[]}

    # count = 0
    #print ('total ', len(temp_json['annotations']))
    for i in temp_json['annotations']:
        # if count %100 ==0:
        #     print (count)
        # count+=1
        if i['category_id'] in cat_ids:
            temp = i
            #print (i, temp)
            temp['category_id'] = id2id[temp['category_id']]
            data['annotations'].append(temp)
            keep_imgs.append(i['image_id'])
    
    #print ('here')
            
    keep_imgs = set(keep_imgs)

    for i in temp_json['categories']:
        if i['id'] in cat_ids:
            temp = i
            temp['id'] = id2id[temp['id']]
            data['categories'].append(temp)
            #data['categories'].append(i)
    
    data['info'] = temp_json['info']
    data['licenses'] = temp_json['licenses']

    #count = 0
    print ('total images:', len(temp_json['images']), '  keeping:', len(keep_imgs))
    for i in temp_json['images']:
        if i['id'] in keep_imgs:
            data['images'].append(i)

    #print ('\n dumping \n')
    # with open(output_dir+'/temp_'+set_type+'.json', 'w') as f:
    #     json.dump(data, f)
    
    with open(os.path.join(output_dir,set_type+'_task_'+str(task_id)+'.json'),'w') as f:
        json.dump(data, f)
    

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--n_tasks', default=2, type=int)
    parser.add_argument('--train_ann', default="/data/coco/annotations/instances_train2017.json", type=str)
    parser.add_argument('--test_ann', default="/data/coco/annotations/instances_val2017.json", type=str)
    parser.add_argument('--output_dir', default="/data/zyt/70", type=str)
    parser.add_argument('--split_point',default=70, type=int)
    
    return parser


def main(args):

    args.output_dir = os.path.join(args.output_dir,str(args.split_point))

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    task_map, task_label2name =  task_info_coco(split_point=args.split_point)

    for task_id in range(1,args.n_tasks+1):
        cur_task = task_map[task_id]

        create_task_json(root_json=args.train_ann,
                        cat_names=cur_task[0], offset=cur_task[1], set_type='train', output_dir=args.output_dir, task_id=task_id)

        create_task_json(root_json=args.test_ann,
                        cat_names=cur_task[0], offset=cur_task[1], set_type='test', output_dir=args.output_dir, task_id=task_id)

    if args.split_point>0:
        create_task_json(root_json=args.test_ann,
                cat_names=task_map[1][0]+task_map[2][0], offset=0, set_type='test', output_dir=args.output_dir, task_id='12')


    for task_id in range(3,args.n_tasks+1):
        cur_task = task_map[task_id]
        known_task_ids = ''.join(str(i) for i in range(1,task_id))
        all_cat_names = []

        for i in range(1,task_id):
            all_cat_names.extend(task_map[i][0])
        
        create_task_json(root_json=args.test_ann,
                        cat_names=all_cat_names, offset=0, set_type='test', output_dir=args.output_dir, task_id=known_task_ids)
    
    known_task_ids = ''.join(str(i) for i in range(1,task_id+1))
    all_cat_names = []

    for i in range(1,task_id+1):
        all_cat_names.extend(task_map[i][0])
    
    create_task_json(root_json=args.test_ann,
                    cat_names=all_cat_names, offset=0, set_type='test', output_dir=args.output_dir, task_id=known_task_ids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

    # task_map, task_label2name =  task_info_coco(split_point=args.split_point)

    # cur_task = task_map[4]
    # known_task_ids = ''.join(str(i) for i in range(1,5))
    # all_cat_names = []

    # for i in range(1,5):
    #     all_cat_names.extend(task_map[i][0])
    
    # create_task_json(root_json=args.test_ann,
    #                 cat_names=all_cat_names, offset=0, set_type='test', output_dir=args.output_dir, task_id=known_task_ids)


    print ('\n Done .... \n')