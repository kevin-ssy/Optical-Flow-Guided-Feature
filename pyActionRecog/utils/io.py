import numpy as np
import math
import cv2


def flow_stack_oversample(flow_stack, crop_dims):
    """
    This function performs oversampling on flow stacks.
    Adapted from pyCaffe's oversample function
    :param flow_stack:
    :param crop_dims:
    :return:
    """
    im_shape = np.array(flow_stack.shape[1:])
    stack_depth = flow_stack.shape[0]
    crop_dims = np.array(crop_dims)

    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])

    h_center_offset = (im_shape[0] - crop_dims[0]) / 2
    w_center_offset = (im_shape[1] - crop_dims[1]) / 2

    crop_ix = np.empty((5, 4), dtype=int)

    cnt = 0
    for i in h_indices:
        for j in w_indices:
            crop_ix[cnt, :] = (i, j, i + crop_dims[0], j + crop_dims[1])
            cnt += 1
    crop_ix[4, :] = [h_center_offset, w_center_offset,
                     h_center_offset + crop_dims[0], w_center_offset + crop_dims[1]]

    crop_ix = np.tile(crop_ix, (2, 1))

    crops = np.empty((10, flow_stack.shape[0], crop_dims[0], crop_dims[1]),
                     dtype=flow_stack.dtype)

    for ix in xrange(10):
        cp = crop_ix[ix]
        crops[ix] = flow_stack[:, cp[0]:cp[2], cp[1]:cp[3]]
    crops[5:] = crops[5:, :, :, ::-1]
    crops[5:, range(0, stack_depth, 2), ...] = 255 - crops[5:, range(0, stack_depth, 2), ...]
    return crops


def rgb_oversample(image, crop_dims):
    """
    Crop images into the four corners, center, and their mirrored versions.
    Adapted from Caffe
    Parameters
    ----------
    image : (H x W x K) ndarray
    crop_dims : (height, width) tuple for the crops.
    Returns
    -------
    crops : (10 x H x W x K) ndarray of crops.
    """
    # Dimensions and center.
    im_shape = np.array(image.shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates
    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
        crop_dims / 2.0
    ])
    crops_ix = np.tile(crops_ix, (2, 1))

    # Extract crops
    crops = np.empty((10, crop_dims[0], crop_dims[1],
                      im_shape[-1]), dtype=np.float32)

    ix = 0
    for crop in crops_ix:
        crops[ix] = image[crop[0]:crop[2], crop[1]:crop[3], :]
        ix += 1
    crops[ix - 5:ix] = crops[ix - 5:ix, :, ::-1, :]  # flip for mirrors
    return crops


def rgb_to_parrots(frame, oversample=True, mean_val=None, crop_size=None):
    """
    Pre-process the rgb frame for Parrots input
    """
    if mean_val is None:
        mean_val = [104, 117, 123]
    if not oversample:
        ret_frame = (frame - mean_val).transpose((2, 0, 1))
        return ret_frame[None, ...]
    else:
        crops = rgb_oversample(frame, crop_size) - mean_val
        ret_frames = crops.transpose((0, 3, 1, 2))
        return ret_frames


def fast_list2arr(data, offset=None, dtype=None):
    """
    Convert a list of numpy arrays with the same size to a large numpy array.
    This is way more efficient than directly using numpy.array()
    See
        https://github.com/obspy/obspy/wiki/Known-Python-Issues
    :param data: [numpy.array]
    :param offset: array to be subtracted from the each array.
    :param dtype: data type
    :return: numpy.array
    """
    num = len(data)
    out_data = np.empty((num,) + data[0].shape, dtype=dtype if dtype else data[0].dtype)
    for i in xrange(num):
        out_data[i] = data[i] - offset if offset else data[i]
    return out_data


def putVecMaps(entryX, count, centerA, centerB, stride, grid_x, grid_y, thre, peak_ratio):
    """

    :param entryX: [row, col]
    :param count: [row, col]
    :param centerA: [row, col]
    :param centerB: [row, col]
    :param stride:
    :param grid_x: num_rows
    :param grid_y: num_cols
    :param thre:
    :param peak_ratio:
    :return:
    """
    centerB = map(lambda x: x * (1 / stride), centerB)
    centerA = map(lambda x: x * (1 / stride), centerA)
    bc = map(lambda x, y: x - y, centerB, centerA)
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)

    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    norm_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
    bc[0] = bc[0] / norm_bc if norm_bc != 0 else bc[0]
    bc[1] = bc[1] / norm_bc if norm_bc != 0 else bc[1]

    for g_y in xrange(min_y, max_y):
        for g_x in xrange(min_x, max_x):

            ba = [g_x - centerA[0], g_y - centerA[1]]
            dist = abs(ba[0] * bc[1] - ba[1] * bc[0])
            if dist <= thre:
                cnt = count[g_x, g_y]
                if cnt == 0:
                    entryX[g_x, g_y] = int(bc[0] * peak_ratio)
                else:
                    entryX[g_x, g_y] = entryX[g_x, g_y] * cnt + int(bc[0]) / (cnt + 1) #bc[0] *
                    count[g_x, g_y] = cnt + 1
    return entryX


def addROI(rois, top_w, top_h, w, h):
    """

    :param rois:
    :param top_w:
    :param top_h:
    :param w:
    :param h:
    :return:
    """
    # rois.push_back(id)
    rois.append(top_w)
    rois.append(top_h)
    rois.append(w)
    rois.append(h)


def calcLineBasedROI(joints, rois, jid0, jid1, roi_h, roi_w):
    is_valid0 = joints[jid0 * 3 + 2]
    is_valid1 = joints[jid1 * 3 + 2]
    if is_valid0 and is_valid1:
        edge = math.sqrt(float(joints[jid0 * 3] - joints[jid1 * 3])
                         * float(joints[jid0 * 3] - joints[jid1 * 3])
                         + float(joints[jid0 * 3 + 1] - joints[jid1 * 3 + 1])
                         * float(joints[jid0 * 3 + 1] - joints[jid1 * 3 + 1]))
        if edge >= min(roi_h, roi_w):
            center = ((joints[jid0 * 3] + joints[jid1 * 3]) / 2,
                      (joints[jid0 * 3 + 1] + joints[jid1 * 3 + 1]) / 2)
            l_w = int(center[0] - edge / 2)
            l_h = int(center[1] - edge / 2)
            egde_to_save = int(edge)
            addROI(rois, l_w, l_h, egde_to_save, egde_to_save)
            return True
    return False



def generateLimb(img_dst, person_joints, scale_x, scale_y):
    rezX = img_dst.shape[1]
    rezY = img_dst.shape[0]
    # print 'rezx: {}'.format(rezX)
    # print 'rezy: {}'.format(rezY)
    mid_1 = [2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16]
    mid_2 = [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]
    thre = 10
    for i in xrange(19):
        count = np.zeros((rezY, rezX))
        for k in xrange(len(person_joints)):
            # print '{} persons in list'.format(len(person_joints))
            joints = person_joints[k]
            is_visible0 = joints[3 * (mid_1[i] - 1) + 2]
            is_visible1 = joints[3 * (mid_2[i] - 1) + 2]
            # print 'is_visible0: {}'.format(is_visible0)
            # print 'is_visible1: {}'.format(is_visible1)
            # print 'point0: {}, {}'.format(joints[3 * (mid_1[i] - 1)] * scale_x, joints[3 * (mid_1[i] - 1) + 1] * scale_y)
            # print 'point1: {}, {}'.format(joints[3 * (mid_2[i] - 1)] * scale_x, joints[3 * (mid_2[i] - 1) + 1] * scale_y)
            if is_visible0 > 0 and is_visible1 > 0:
                img_dst = putVecMaps(img_dst, count,
                                     (joints[3 * (mid_1[i] - 1)] * scale_x, joints[3 * (mid_1[i] - 1) + 1] * scale_y),
                                     (joints[3 * (mid_2[i] - 1)] * scale_x, joints[3 * (mid_2[i] - 1) + 1] * scale_y),
                                     1.0, rezY, rezX, thre, 255.0)
    img_dst = img_dst.transpose([1, 0]) # OpenCV col/row regulation transforming
    return img_dst


def generateROI(person_joints, select_joints, scale_x, scale_y, roi_h, roi_w, roi_mult=2.4):
    num_select_joints = len(select_joints)
    rois = []
    top_w, top_h, w, h = 0, 0, 0, 0

    for k in xrange(len(person_joints)):
        joints = person_joints[k]
        if num_select_joints == 2:
            if calcLineBasedROI(joints, rois, select_joints[0], select_joints[1], roi_h, roi_w):
                continue
    # TODO: arms & wrists

    # general case, generating bounding box from given key points
    for i in xrange(num_select_joints):
        jid = select_joints[i]
        is_valid = joints[3 * jid + 2]
        if is_valid > 0:
            x = int(joints[3 * jid] * scale_x)
            y = int(joints[3 * jid + 1] * scale_y)
            if x < 0 or y < 0:
                continue

            top_w = x if top_w == 0 and x > 0 else x if x < top_w else top_w
            top_h = y if top_h == 0 and y > 0 else y if y < top_h else top_h
            if top_h or top_w:
                w = x - top_w if w == 0 and x - top_w > 0 else x - top_w if x - top_w > w else w
                h = y - top_h if h == 0 and y - top_h > 0 else y - top_h if y - top_h > h else h
            if (top_h or top_w or w or h) and (w > roi_w or h > roi_h):
                e = int(max(w, h))
                if 17 in select_joints and joints[3 * 17 + 2] > 0:
                    top_w = int(joints[3 * 17] * scale_x - e / 2) if int(joints[3 * 17] * scale_x - e / 2) >= 0 else 0
                    top_h = int(joints[3 * 17 + 1] * scale_y - e / 2) if int(joints[3 * 17 + 1] * scale_y - e / 2) >= 0 else 0
                addROI(rois, top_w, top_h, e, e)
    if rois:
        select_roi = rois[0]
    else:
        return 0,0,0,0
    datum_top_w, datum_top_h, datum_roi_w, datum_roi_h = rois[0], rois[1], rois[2], rois[3]
    intersect_top = [max(datum_top_w, 0), max(datum_top_h, 0)]
    intersect_bot = [min(datum_top_w + datum_roi_w, 340), min(datum_top_h + datum_roi_h, 256)]

    roi_top_w, roi_top_h, roi_w, roi_h = 0, 0, 0, 0
    if (intersect_top[0] - intersect_bot[0] <= 0 and intersect_top[1] - intersect_bot[1] <= 0):
        roi_top_w = (intersect_top[0]) - (datum_roi_w * roi_mult - datum_roi_w) / 2 if \
            (intersect_top[0]) - (datum_roi_w * roi_mult - datum_roi_w) / 2 > 0 else 0
        roi_top_h = (intersect_top[1]) - (datum_roi_h * roi_mult - datum_roi_h) / 2 if \
            (intersect_top[1]) - (datum_roi_h * roi_mult - datum_roi_h) / 2 > 0 else 0
        roi_w = 340 - roi_top_w if (intersect_bot[0] - intersect_top[0]) * roi_mult > 340 - roi_top_w else \
            (intersect_bot[0] - intersect_top[0]) * roi_mult
        roi_h = 256 - roi_top_h if (intersect_bot[1] - intersect_top[1]) * roi_mult > 256 - roi_top_h else \
            (intersect_bot[1] - intersect_top[1]) * roi_mult
    return int(roi_top_w), int(roi_top_h), int(roi_w), int(roi_h)