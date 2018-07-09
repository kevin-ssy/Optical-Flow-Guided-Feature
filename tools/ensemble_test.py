from pyActionRecog.utils.video_funcs import default_aggregation_func
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix


SPLIT = '1'

with open('final_scores/ucf101_rgb_off_reference_split{}.pickle'.format(SPLIT)) as fmotion:
    print 'loading rgb off...'
    motion_score = pickle.load(fmotion)

with open('final_scores/ucf101_flow_off_reference_split{}.pickle'.format(SPLIT)) as fmotion:
    print 'loading flow off...'
    flow_score = pickle.load(fmotion)

with open('final_scores/ucf101_rgb_diff_off_reference_split{}.pickle'.format(SPLIT)) as fmotion:
    print 'loading rgb_diff off...'
    diff_score = pickle.load(fmotion)


aligned_flow_score = [None for _ in xrange(len(flow_score))]
for mid, motion_sample in enumerate(motion_score):
    for fid, flow_sample in enumerate(flow_score):
        if motion_sample[2] == flow_sample[2]:
            aligned_flow_score[mid] = flow_sample

# RGB : RGB OFF : FLOW : FLOW OFF -- 1:1.5:0.8:1.8
# RGB : RGB OFF : (RGB DIFF : RGB DIFF OFF : RGB DIFF OFF 14)= 1:1.8:  (1:2:0.5)*0.8
# 0~25: Scores from Feature Generation Network
# 25~49: Score from OFF-sub-network on 7x7
# 49~73: Score from OFF-sub-network on 14x14
video_pred = [np.argmax(
    default_aggregation_func(x[0][:25, ...], normalization=False, crop_agg=np.max) * 1 +
    default_aggregation_func(x[0][25:49, ...], normalization=False, crop_agg=np.max) * 1.5

   + default_aggregation_func(y[0][:25, ...], normalization=False, crop_agg=np.max) * 0.8 
   +  default_aggregation_func(y[0][25:49, ...], normalization=False, crop_agg=np.max) * 1.8
)
   for x,y,z in zip(motion_score, aligned_flow_score, diff_score)]

video_labels = [x[1] for x in motion_score]
video_names = [x[2] for x in motion_score]

cf = confusion_matrix(video_labels, video_pred).astype(float)
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)
cls_acc = cls_hit / cls_cnt

print cls_acc
print 'Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100)
