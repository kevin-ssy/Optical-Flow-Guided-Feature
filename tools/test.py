"""
Test code for Optical-Flow-Guided-Feature with multi-gpu support.
@author: Shuyang Sun, the University of Sydney.

"""
import argparse
import os
import sys
import cv2
from math import floor
import numpy as np
from sklearn.metrics import confusion_matrix
import multiprocessing
from pyActionRecog.utils.video_funcs import default_aggregation_func
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('net_proto_template', type=str)
parser.add_argument('trunk_net_base', type=str)
parser.add_argument('net_weights', type=str)
parser.add_argument('source_path', type=str)
parser.add_argument('--save_scores', type=str, default=None, help='the filename to save the scores in')
parser.add_argument('--num_worker', type=int, default=1)
parser.add_argument("--caffe_path", type=str, default='./lib/caffe-action/', help='path to the caffe toolbox')
parser.add_argument("--check_result", type=bool, default=True, help='Whether check the result')

args = parser.parse_args()

sys.path.append(os.path.join(args.caffe_path, 'python'))
from pyActionRecog.action_caffe import CaffeNet

gpu_list = list(xrange(args.num_worker))


def write_split(line, split_id, net_bsae):
    # split_id = multiprocessing.current_process()._identity[0]
    with open('{}/source_{}'.format(net_bsae, split_id), 'a+') as fdest:
        fdest.write(line)


def split_source(source_path, net_base, num_splits=4):
    with open(source_path) as fsource:
        lines = fsource.readlines()
    seg_size = int(floor(len(lines) / num_splits))
    for split_id in xrange(num_splits):
        if not split_id == num_splits - 1:
            [write_split(line, split_id, net_base) for line in lines[seg_size*split_id:seg_size*(split_id+1)]]
        else:
            [write_split(line, split_id, net_base) for line in lines[seg_size * split_id:]]


def gen_param_list(net_base, num_splits=4):
    global trunk_net_base
    param_list = []
    for split_id in xrange(num_splits):
        param_list.append({'$SOURCE': '\"{}/source_{}\"'.format(trunk_net_base, split_id),
                           '$SCALE_MULT_PATH': '\"{}/scale_mult_{}\"'.format(net_base, split_id),
                           '$OVERSAMPLE_ID_PATH': '\"{}/oversample_id_{}\"'.format(net_base, split_id),
                           '$VIDEO_ID_PATH': '\"{}/video_id_{}\"'.format(net_base, split_id)})
        # '$PROCESS_ID': '{}'.format(split_id + 1),
    return param_list


def write_proto_template(template_proto_path, num_worker, net_base):
    """
    [{'$SOURCE': source_split1},
     {'$SOURCE': source_split2},
     {'$SOURCE': source_split3},
     {'$SOURCE': source_split4}]
    :param param_list:
    :return:
    """
    param_list = gen_param_list(net_base, num_worker)
    proto_temp = ''
    with open(template_proto_path) as ftemplate:
        proto_temp = ftemplate.read()
    for split_id, split_param in enumerate(param_list):
        assert isinstance(split_param, dict)
        ket_set = split_param.keys()
        proto_split = proto_temp
        for key in ket_set:
            proto_split = proto_split.replace(key, split_param[key])
        with open('{}/deploy_{}.prototxt'.format(net_base, split_id), 'w') as fsplit:
            fsplit.write(proto_split)


def eval_video(gpu_id):
    global trunk_net_base
    print 'GPU %d is processing' % gpu_id
    score_name_list = ['fc', 'fc_motion', 'fc_motion_14']

    # initialization
    net = CaffeNet('{}/deploy_{}.prototxt'.format(trunk_net_base, gpu_id), args.net_weights, gpu_id)

    # preparing video names
    video_names = []
    with open('{}/source_{}'.format(trunk_net_base, gpu_id)) as fsource_split:
        video_names = map(lambda l: os.path.basename(l.split(' ')[0]), fsource_split.readlines())
        length_split = len(video_names)
    video_result = []

    for video_id in xrange(length_split):
        with open('{}/video_id_{}'.format(trunk_net_base, gpu_id), 'w') as fvid:
            fvid.write('{}'.format(video_id))

        scores, label = net.predict_single_frame_motion(trunk_net_base, score_name_list, gpu_id, over_sample=True)

        print '[Worker {}] video {}: {} Done'.format(gpu_id, video_id, video_names[video_id])
        video_result.append((scores, label, video_names[video_id]))
    return video_result


def merge_worker_result(raw_data):
    merged_result = []
    [merged_result.extend(result_batch) for result_batch in raw_data]
    return merged_result


if __name__ == '__main__':
    # load parameters from outer space
    # dest_source_path = args.source_split_base
    trunk_net_base = args.trunk_net_base
    num_worker = args.num_worker

    split_source(args.source_path, trunk_net_base, num_worker)
    write_proto_template(args.net_proto_template, num_worker, trunk_net_base)
    if len(gpu_list) > 1:
        cnn_worker = multiprocessing.Pool(len(gpu_list))
        raw_video_scores = cnn_worker.map(eval_video, gpu_list)
        video_scores = merge_worker_result(raw_video_scores)
    else:
        video_scores = eval_video(0)

    with open('{}.pickle'.format(args.save_scores), 'w') as fv_score:
        pickle.dump(video_scores, fv_score)

    video_pred = [np.argmax(default_aggregation_func(x[0], crop_agg=np.max)) for x in video_scores]
    video_labels = [x[1] for x in video_scores]
    video_names = [x[2] for x in video_scores]

    cf = confusion_matrix(video_labels, video_pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt

    print cls_acc
    if args.check_result:
        cnt = 0
        for vid, video_label in enumerate(video_labels):
            if video_label != video_pred[vid]:
                print video_names[vid]
            else:
                cnt += 1
        print 'Overall Accuracy: {:.02f}%'.format((cnt / len(video_pred)) * 100)
    print 'Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100)
