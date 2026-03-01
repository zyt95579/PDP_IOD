_all_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def task_info_voc_10(split_point=0):
    T1_CLASS_NAMES = _all_classes[:10]
    T2_CLASS_NAMES = _all_classes[10:]

    all_classes = T1_CLASS_NAMES + T2_CLASS_NAMES
    task_map = {1: (T1_CLASS_NAMES, 0, 10), 2: (T2_CLASS_NAMES, 10, 10), }
    task_label2name = {}
    for i, j in enumerate(all_classes):
        task_label2name[i] = j
    return task_map, task_label2name

def task_info_voc_15(split_point=0):
    T1_CLASS_NAMES = _all_classes[:15]
    T2_CLASS_NAMES = _all_classes[15:]

    all_classes = T1_CLASS_NAMES + T2_CLASS_NAMES
    task_map = {1: (T1_CLASS_NAMES, 0, 15), 2: (T2_CLASS_NAMES, 15, 5), }
    task_label2name = {}
    for i, j in enumerate(all_classes):
        task_label2name[i] = j
    return task_map, task_label2name

def task_info_voc_19(split_point=0):
    T1_CLASS_NAMES = _all_classes[:19]
    T2_CLASS_NAMES = _all_classes[19:]

    all_classes = T1_CLASS_NAMES + T2_CLASS_NAMES
    task_map = {1: (T1_CLASS_NAMES, 0, 19), 2: (T2_CLASS_NAMES, 19, 1), }
    task_label2name = {}
    for i, j in enumerate(all_classes):
        task_label2name[i] = j
    return task_map, task_label2name