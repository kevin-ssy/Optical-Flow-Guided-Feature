import scipy.io as sio
import numpy as np
import os
import cv2


def build_file_list(root_dir, white_list):
    file_tree = []
    root_list = []
    build_file_tree(root_dir, file_tree, white_list)
    print 'Parsing {}...'.format(root_dir)
    [root_list.extend(json_folder[:]) for json_folder in file_tree if isinstance(json_folder, list)]
    return root_list


def build_file_tree(root_dir, root_list, white_list):
    """
    build file tree in a regression way
    :param root_dir:
    :param root_list:
    :param white_list:
    :return:
    """
    root_dir_list = os.listdir(root_dir)
    flist = []
    for sub_dir_path in root_dir_list:
        if sub_dir_path in white_list:
            continue
        new_root = root_dir + sub_dir_path
        flag = os.path.isdir(new_root)
        if flag:
            build_file_tree(new_root + '/', root_list, white_list)
        else:
            flist.append(new_root)
    print 'Parsing {}...'.format(root_dir)
    root_list.append(flist)


def is_valid_arr(arr):
    assert isinstance(arr, np.ndarray)
    for x in arr:
        if x[0] != -1:
            return True
    return False


def get_valid_list(subset):
    assert isinstance(subset, np.ndarray)
    valid_list = []
    for pid in xrange(subset.shape[0]):
        if is_valid_arr(subset[pid, :, :]):
            valid_list.append(pid)
    return valid_list


def parse_mat(mat_path):
    fmat = sio.loadmat(mat_path)
    person_id_list = get_valid_list(fmat['subset'])
    landmark_id_list = get_valid_list(fmat['landmark'])
    num_valid_person = len(person_id_list)
    person_list = []
    landmark_list = []
    [person_list.extend(fmat['subset'][pid, :, :].tolist()) for pid in person_id_list]
    [landmark_list.extend(fmat['landmark'][lid, :, :].tolist()) for lid in landmark_id_list]
    person_list = map(lambda x: x[0], person_list)
    landmark_list = map(lambda x: x[0], landmark_list)

    return person_list, landmark_list, num_valid_person


def filter_points(mat_path):
    data = sio.loadmat(mat_path)
    # image_path = '/home/ethan/pose_tmp/18_baidujpg'
    # data = sio.loadmat(image_path + '.mat')
    candidate = data['landmark']
    subset = data['subset']


    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
               [1, 16], [16, 18], [3, 17], [6, 18]]
    joints = []
    for s in range(len(subset)):  # each line represents for a valid person
        if subset[s][19][0] < 3.5:  # num of valid joints
            continue
        person_joints = []
        [person_joints.append([-1, -1, -1]) for t in xrange(18)]
        for i in range(17):
            index1 = int(subset[s][limbSeq[i][0] - 1][0])
            index2 = int(subset[s][limbSeq[i][1] - 1][0])
            if index1 > -0.2:
                jid = int(candidate[index1][3])
                person_joints[jid][0] = int(candidate[index1][0])
                person_joints[jid][1] = int(candidate[index1][1])
                person_joints[jid][2] = 1
            if index2 > -0.2:
                jid = int(candidate[index2][3])
                person_joints[jid][0] = int(candidate[index2][0])
                person_joints[jid][1] = int(candidate[index2][1])
                person_joints[jid][2] = 1
        joints.append(person_joints)
    return joints


def write_mat_into_txt(fname, person_list, landmark_list, num_valid_person):
    with open(fname, 'a+') as flabel:
        person_str = reduce(lambda x, y: str(x) + ' ' + str(y), person_list)
        landmark_str = reduce(lambda x, y: str(x) + ' ' + str(y), landmark_list)
        flabel.write('{} {} , {}\n'.format(num_valid_person, person_str, landmark_str))


def write_joints(fname, joints, img_path):
    num_person = len(joints)
    for n in xrange(num_person):
        person_str = ''
        for jid, joint in enumerate(joints[n]):
            joint_str = reduce(lambda x, y: str(x) + ' ' + str(y), joint)
            person_str = person_str + ' ' + joint_str
        with open(fname, 'a+') as flabel:
            flabel.write('{} {} {}\n'.format(img_path, num_person, person_str))


def process_mat(mat_path, fname, img_root):
    # person_list, landmark_list, num_valid_person = parse_mat(mat_path)
    # write_mat_into_txt(fname, person_list, landmark_list, num_valid_person)
    if 'v_FrontCrawl_g13_c01' in mat_path:
        pass
    joints = filter_points(mat_path)
    img_dir_name = os.path.basename(os.path.dirname(mat_path))
    img_name = os.path.basename(mat_path)[:-4]
    img_path = os.path.join(img_root, img_dir_name, img_name)

    write_joints(fname, joints, img_path)



            # if index1 > -0.2 and index2 > -0.2:
            #     X1 = int(candidate[index1][0])
            #     Y1 = int(candidate[index1][1])
            #     X2 = int(candidate[index2][0])
            #     Y2 = int(candidate[index2][1])


if __name__ == '__main__':
    file_list = build_file_list('/home/kevin/ori_tsn/joint_result/', [])
    for file_path in file_list:
        process_mat(file_path, 'ucf_keypoints', '/home/kevin/ori_tsn/frames/ucf101/')
    # process_mat('img_00349.jpg.mat', 'ucf_keypoints_list.txt')





