import sys

import caffe
from caffe.io import oversample
import numpy as np
from utils.io import flow_stack_oversample, fast_list2arr, generateLimb, generateROI
import cv2
import random
import pickle


class CaffeNet(object):

    def __init__(self, net_proto, net_weights, device_id, input_size=None):
        caffe.set_mode_gpu()
        caffe.set_device(device_id)
        print '1'
        self._net = caffe.Net(net_proto, net_weights, caffe.TEST)
        input_shape = self._net.blobs['data'].data.shape

        if input_size is not None:
            input_shape = input_shape[:2] + input_size
        print input_shape
        transformer = caffe.io.Transformer({'data': input_shape})

        if self._net.blobs['data'].data.shape[1] == 3:
            transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
            transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
        elif self._net.blobs['data'].data.shape[1] == 4:
            transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
            transformer.set_mean('data', np.array([104, 117, 123, 0]))  # subtract the dataset-mean value in each channel
        else:
            pass # non RGB data need not use transformer

        self._transformer = transformer

        self._sample_shape = self._net.blobs['data'].data.shape

    def predict_single_frame(self, frame, score_name, over_sample=True, multiscale=None, frame_size=None):
        img_id = random.randint(0, 1000)

        if frame_size is not None:
            frame = [cv2.resize(x, frame_size) for x in frame]

        if over_sample:
            if multiscale is None:
                os_frame = oversample(frame, (self._sample_shape[2], self._sample_shape[3]))
            else:
                os_frame = []
                for scale in multiscale:
                    resized_frame = [cv2.resize(x, (0,0), fx=1.0/scale, fy=1.0/scale) for x in frame]
                    os_frame.extend(oversample(resized_frame, (self._sample_shape[2], self._sample_shape[3])))
        else:
            os_frame = fast_list2arr(frame)
        data = fast_list2arr([self._transformer.preprocess('data', x) for x in os_frame])

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name,], data=data)
        return out[score_name].copy()

    def predict_single_flow_stack(self, frame, score_name, over_sample=True, frame_size=None):

        if frame_size is not None:
            frame = fast_list2arr([cv2.resize(x, frame_size) for x in frame])
        else:
            frame = fast_list2arr(frame)

        if over_sample:
            os_frame = flow_stack_oversample(frame, (self._sample_shape[2], self._sample_shape[3]))
        else:
            os_frame = fast_list2arr([frame])

        data = os_frame - np.float32(128.0)

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name,], data=data)
        return out[score_name].copy()

    def predict_single_frame_with_attention(self, frame, score_name, joints,
                                            over_sample=True, multiscale=None, frame_size=None):
        # TODO: uncomment the following to visualize
        # img_id = random.randint(0, 1000)
        # cv2.imwrite('visualize/{}_ori_img.jpg'.format(img_id), frame[0])

        if frame_size is not None:
            frame = [cv2.resize(x, frame_size) for x in frame]
            pose_map = np.zeros(frame_size, dtype='float32')
            scale_x = pose_map.shape[0] / 255. # row
            scale_y = pose_map.shape[1] / 255. # col
            pose_map = [np.expand_dims(generateLimb(pose_map, joints, scale_x, scale_y), axis=2), ]
            # TODO: uncomment the following to visualize
            # cv2.imwrite('visualize/{}_ori_img.jpg'.format(img_id), frame[0])
            # cv2.imwrite('visualize/{}_ori_pose.jpg'.format(img_id), pose_map[0])
            # img_grey_ori = cv2.cvtColor(frame[0], cv2.COLOR_BGRA2GRAY)
            # pose_concat = np.tile(pose_map[0].astype('uint8'), 3)
            # pose_squeezed = pose_map[0].astype('uint8').squeeze(axis=2)
            # pose_color_map = cv2.applyColorMap(pose_concat, cv2.COLORMAP_JET)
            # img_merge_ori = cv2.addWeighted(frame[0], 0.5, pose_color_map, 0.5, 0)
            # cv2.imwrite('visualize/{}_ori_weighted.jpg'.format(img_id), img_merge_ori)

        if over_sample:
            if multiscale is None:
                os_frame = oversample(frame, (self._sample_shape[2], self._sample_shape[3]))
                os_pose_map = oversample(pose_map, (self._sample_shape[2], self._sample_shape[3]))
            else:
                os_frame = []
                os_pose_map = []
                for scale in multiscale:
                    resized_frame = [cv2.resize(x, (0,0), fx=1.0/scale, fy=1.0/scale) for x in frame]
                    resized_pose_map = [cv2.resize(x, (0,0), fx=1.0/scale, fy=1.0/scale) for x in pose_map]
                    os_frame.extend(oversample(resized_frame, (self._sample_shape[2], self._sample_shape[3])))
                    os_pose_map.extend(oversample(resized_pose_map, (self._sample_shape[2], self._sample_shape[3])))
        else:
            os_frame = fast_list2arr(frame)
            os_pose_map = fast_list2arr(pose_map)
        # TODO: uncomment the following to visualize
        # for i in xrange(os_frame.shape[0]):
        #     img_to_show_ = os_frame[i, :, :, :].squeeze()
        #     pose_to_show_ = os_pose_map[i, :, :, :].squeeze()
        #
        #     img_grey_ori_ = cv2.cvtColor(img_to_show_, cv2.COLOR_BGRA2GRAY).astype('uint8')
        #     pose_squeezed_ = pose_to_show_.astype('uint8')
        #     img_merge_ori = cv2.addWeighted(img_grey_ori_, 0.5, pose_squeezed_, 0.5, 0)
        #     cv2.imwrite('visualize/{}_{}_weighted.jpg'.format(img_id, i), img_merge_ori)
        #     cv2.imwrite('visualize/{}_{}_img.jpg'.format(img_id, i), img_to_show_)
        #     cv2.imwrite('visualize/{}_{}_pose.jpg'.format(img_id, i), pose_to_show_)
        raw_data = np.append(os_frame, os_pose_map, axis=3)
        #####################################################################

        data = fast_list2arr([self._transformer.preprocess('data', x) for x in raw_data])
        # TODO: uncomment the following to visualize
        # for i in xrange(os_frame.shape[0]):
        #     img_to_show = data[i, :3, :, :].squeeze().transpose(1, 2, 0)
        #     pose_to_show = data[i, 3, :, :].squeeze()
        #     img_to_show[:, :, 0] += 104
        #     img_to_show[:, :, 1] += 117
        #     img_to_show[:, :, 2] += 123
        #
        #     print img_to_show.shape
        #     print pose_to_show.shape
        #     img_grey_ori = cv2.cvtColor(img_to_show, cv2.COLOR_BGRA2GRAY).astype('uint8')
        #     pose_squeezed = pose_to_show.astype('uint8')
        #     img_merge_ori = cv2.addWeighted(img_grey_ori, 0.5, pose_squeezed, 0.5, 0)
        #     cv2.imwrite('visualize/{}_{}_weighted_post.jpg'.format(img_id, i), img_merge_ori)
        #     cv2.imwrite('visualize/{}_{}_img_post.jpg'.format(img_id, i), img_to_show)
        #     cv2.imwrite('visualize/{}_{}_pose_post.jpg'.format(img_id, i), pose_to_show)
        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name,], data=data)
        print out.max()
        return out[score_name].copy()

    def predict_single_frame_with_roi(self, frame, score_name, joints,
                                            over_sample=True, multiscale=None, frame_size=None):
        # TODO: uncomment the following to visualize
        # img_id = random.randint(0, 1000)
        # cv2.imwrite('visualize/{}_ori_img.jpg'.format(img_id), frame[0])

        assert isinstance(frame_size, tuple)
        frame = [cv2.resize(x, frame_size) for x in frame]
        use_roi = False
        scale_x = frame_size[0] / 336.  # row
        scale_y = frame_size[1] / 256.  # col
        if joints:
            roi_top_w, roi_top_h, roi_w, roi_h = generateROI(joints, [0, 13, 14, 15, 16, 17], scale_x, scale_y, 40, 40)
            if roi_h > 40 and roi_w > 40:
                use_roi = True

            # TODO: uncomment the following to visualize
            # cv2.imwrite('visualize/{}_ori_img.jpg'.format(img_id), frame[0])
            # cv2.imwrite('visualize/{}_ori_pose.jpg'.format(img_id), pose_map[0])
            # img_grey_ori = cv2.cvtColor(frame[0], cv2.COLOR_BGRA2GRAY)
            # pose_concat = np.tile(pose_map[0].astype('uint8'), 3)
            # pose_squeezed = pose_map[0].astype('uint8').squeeze(axis=2)
            # pose_color_map = cv2.applyColorMap(pose_concat, cv2.COLORMAP_JET)
            # img_merge_ori = cv2.addWeighted(frame[0], 0.5, pose_color_map, 0.5, 0)
            # cv2.imwrite('visualize/{}_ori_weighted.jpg'.format(img_id), img_merge_ori)

        if over_sample:
            if multiscale is None and not use_roi:
                os_frame = oversample(frame, (self._sample_shape[2], self._sample_shape[3]))
            elif use_roi:
                os_frame = []
                roi_mult_list = np.arange(2., 3., 0.1).tolist()
                for roi_mult in roi_mult_list:
                    roi_top_w, roi_top_h, roi_w, roi_h = generateROI(joints, [0, 13, 14, 15, 16, 17], scale_x, scale_y,
                                                                     40, 40, roi_mult)
                    target_size = (self._sample_shape[2], self._sample_shape[3])
                    resized_roi = [cv2.resize(x[roi_top_h:roi_h + roi_top_h, roi_top_w:roi_w + roi_top_w],
                                              target_size) for x in frame]
                    os_frame.extend(resized_roi)
            else:
                os_frame = []
                for scale in multiscale:
                    resized_frame = [cv2.resize(x, (0, 0), fx=1.0/scale, fy=1.0/scale) for x in frame]
                    os_frame.extend(oversample(resized_frame, (self._sample_shape[2], self._sample_shape[3])))
        else:
            os_frame = fast_list2arr(frame)
        # TODO: uncomment the following to visualize
        # for i in xrange(len(os_frame)):
        #     img_to_show_ = os_frame[i].squeeze()
        #     cv2.imwrite('visualize/{}_{}_img.jpg'.format(img_id, i), img_to_show_)
        #     pose_to_show_ = os_pose_map[i, :, :, :].squeeze()
        #
        #     img_grey_ori_ = cv2.cvtColor(img_to_show_, cv2.COLOR_BGRA2GRAY).astype('uint8')
        #     pose_squeezed_ = pose_to_show_.astype('uint8')
        #     img_merge_ori = cv2.addWeighted(img_grey_ori_, 0.5, pose_squeezed_, 0.5, 0)
        #     cv2.imwrite('visualize/{}_{}_weighted.jpg'.format(img_id, i), img_merge_ori)
        #     cv2.imwrite('visualize/{}_{}_img.jpg'.format(img_id, i), img_to_show_)
        #     cv2.imwrite('visualize/{}_{}_pose.jpg'.format(img_id, i), pose_to_show_)
        # raw_data = np.append(os_frame, os_pose_map, axis=3)
        #####################################################################

        data = fast_list2arr([self._transformer.preprocess('data', x) for x in os_frame])
        # TODO: uncomment the following to visualize
        # for i in xrange(os_frame.shape[0]):
        #     img_to_show = data[i, :3, :, :].squeeze().transpose(1, 2, 0)
        #     pose_to_show = data[i, 3, :, :].squeeze()
        #     img_to_show[:, :, 0] += 104
        #     img_to_show[:, :, 1] += 117
        #     img_to_show[:, :, 2] += 123
        #
        #     print img_to_show.shape
        #     print pose_to_show.shape
        #     img_grey_ori = cv2.cvtColor(img_to_show, cv2.COLOR_BGRA2GRAY).astype('uint8')
        #     pose_squeezed = pose_to_show.astype('uint8')
        #     img_merge_ori = cv2.addWeighted(img_grey_ori, 0.5, pose_squeezed, 0.5, 0)
        #     cv2.imwrite('visualize/{}_{}_weighted_post.jpg'.format(img_id, i), img_merge_ori)
        #     cv2.imwrite('visualize/{}_{}_img_post.jpg'.format(img_id, i), img_to_show)
        #     cv2.imwrite('visualize/{}_{}_pose_post.jpg'.format(img_id, i), pose_to_show)
        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name,], data=data)
        # print np.argmax(out[score_name], axis=1)
        # TODO: check wrong samples

        return out[score_name].copy()

    def get_result(self, result, out, score_name):
        if result is None:
            result = out[score_name]
            result = np.expand_dims(result, axis=0)
        else:
            result = np.append(result, np.expand_dims(out[score_name].copy(), axis=0), axis=0)
        return result

    def get_score_label(self, label_name):
        out = self._net.forward()
        label = out[label_name]
        return out, label

    def forward_roi_net(self, net_base, pid, roi_mult):
        with open('{}/scale_mult_{}'.format(net_base, pid), 'w') as fscale:
            fscale.write('{:.1f}'.format(roi_mult))
        out, label = self.get_score_label('label')
        return out, label

    def forward_rgb_net(self, net_base, pid, os_id):
        with open('{}/oversample_id_{}'.format(net_base, pid), 'w') as fscale:
            fscale.write('{}'.format(os_id))
        out, label = self.get_score_label('label')
        return out, label

    def forward_merge_net(self, net_base, pid, os_id, roi_mult):
        with open('{}/scale_mult_{}'.format(net_base, pid), 'w') as fscale:
            fscale.write('{:.1f}'.format(roi_mult))
        with open('{}/oversample_id_{}'.format(net_base, pid), 'w') as fscale:
            fscale.write('{}'.format(os_id))
        out, label = self.get_score_label('label')
        return out, label

    def predict_single_frame_from_cpp(self, net_base, frame_name, score_name, pid, is_roi=True, over_sample=True,
                                      save_score=False, is_merge=False):
        result = None
        label = 0
        if is_roi:
            if over_sample:
                roi_mult_list = np.arange(2.2, 3.2, 0.1).tolist()
                for roi_mult in roi_mult_list:
                    out, label = self.forward_roi_net(net_base, pid, roi_mult)
                    result = self.get_result(result, out, score_name)
            else:
                roi_mult = 2.5  # np.arange(2., 3., 0.1).tolist()
                out, label = self.forward_roi_net(net_base, pid, roi_mult)
                result = self.get_result(result, out, score_name)
        elif is_merge:
            if over_sample:
                oversample_id_list = np.arange(0, 10).tolist()
                roi_mult_list = np.arange(2.4, 3.4, 0.1).tolist()
                for os_id, roi_mult in zip(oversample_id_list, roi_mult_list):
                    out, label = self.forward_merge_net(net_base, pid, os_id, roi_mult)
                    result = self.get_result(result, out, score_name)
            else:
                roi_mult = 2.5
                out, label = self.forward_roi_net(net_base, pid, roi_mult)
                result = self.get_result(result, out, score_name)
        else:  # trunk
            if over_sample:
                oversample_id_list = np.arange(0, 10).tolist()
                for os_id in oversample_id_list:
                    out, label = self.forward_rgb_net(net_base, pid, os_id)
                    result = self.get_result(result, out, score_name)
            else:
                out = self._net.forward()  # blobs=[score_name, ], data=data
                label = out['label']
                result = self.get_result(result, out, score_name)
        #####################################################################
        if save_score:
            with open('scores/{}.pkl'.format(frame_name), 'w') as fscore:
                pickle.dump(result, fscore)
        # print np.argmax(out[score_name], axis=1)

        return np.swapaxes(result, 0, 1).copy(), int(label.max())

    def predict_single_frame_motion(self, net_base, fc_score_name_list, pid, over_sample=True):
        # result_fc = None
        # result_fusion = None
        result_fc_dict = {}
        for fc_score_name in fc_score_name_list:
            result_fc_dict[fc_score_name] = None
        result = None
        label = 0
        if over_sample:
            oversample_id_list = np.arange(0, 10).tolist()
            for os_id in oversample_id_list:
                with open('{}/oversample_id_{}'.format(net_base, pid), 'w') as fscale:
                    fscale.write('{}'.format(os_id))
                out = self._net.forward()  # blobs=[score_name, ], data=data
                label = out['label']
                for fc_score_name in fc_score_name_list:
                    result_fc_dict[fc_score_name] = self.get_result(result_fc_dict[fc_score_name], out, fc_score_name)
        else:
            out = self._net.forward()  # blobs=[score_name, ], data=data
            label = out['label']
            for fc_score_name in fc_score_name_list:
                result_fc_dict[fc_score_name] = self.get_result(result_fc_dict[fc_score_name], out, fc_score_name)
        for fc_score_name in fc_score_name_list:
            if result is None:
                result = result_fc_dict[fc_score_name]
            else:
                result = np.append(result, result_fc_dict[fc_score_name], axis=1)

        return np.swapaxes(result, 0, 1).copy(), int(label.max())

