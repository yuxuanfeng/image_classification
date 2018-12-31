import sys
import os
import time
import subprocess as sp
import deep_feature as df
import numpy as np

raw_image_set = './real_test'
new_image_set = './real_set_image'
feature_set = './real_set_feature'
test_file = './real_image_feature'

image_list = './real_image_list'
image_type_list = ['png', 'jpg', 'jpeg']
discard_file_name = 'DS_Store'

is_run_step1 = True
is_run_step2 = True
is_run_step3 = True
is_run_step4 = True

MY_ZERO_VALUE = 1e-5


def prepare_workspace():
    if not os.path.isdir(new_image_set):
        os.mkdir(new_image_set)

    if not os.path.isdir(feature_set):
        os.mkdir(feature_set)

    if not os.path.isfile(image_list):
        cmd_str = 'touch ' + image_list
        if sp.call(cmd_str, shell=True) != 0:
            print('Fail to run ' + cmd_str)
            sys.exit(-1)


def image_class_real():
    print('image class recognition demo ... \n')
    prepare_workspace()

    print('[step-1] change Chinese name into English name...')
    if is_run_step1:
        change_image_name(raw_image_set, new_image_set)

    print('[step-2] get image list...')
    if is_run_step2:
        get_image_list(new_image_set, image_list)

    print('[step-3] image feature extraction....')
    if is_run_step3:
        get_image_feature(image_list, feature_set)

    print('[step-4] test image classification acc')
    if is_run_step4:
        test_algo_model(image_list, feature_set, test_file)


def get_image_list(folder_name, image_list):
    pid = os.walk(folder_name)

    with open(image_list, 'w') as fid:
        for path, dir_list, file_list in pid:
            for file_name in file_list:
                if discard_file_name not in file_name:
                    full_name = os.path.join(path, file_name)
                    print('image name:' + full_name)
                    fid.write(full_name + '\n')


def change_image_name(org_image_set, tag_image_set):
    folder_name_dict = {}
    folder_name_count = 0
    pid = os.walk(org_image_set)

    for path, dir_list, file_list in pid:
        for folder_name in dir_list:
            if folder_name not in folder_name_dict:
                folder_name_dict[folder_name] = folder_name_count
                folder_name_count += 1
        file_name_count = 0
        for file_name in file_list:
            file_name = file_name.strip('\n')
            if discard_file_name not in file_name:
                raw_full_name = os.path.join(path, file_name)
                sub_fields = raw_full_name.split('/')
                new_path = tag_image_set + '/class_' + str(folder_name_dict[sub_fields[2]])
                new_name = 'image_' + str(file_name_count) + file_name[file_name.rfind('.'):]
                new_full_name = os.path.join(new_path, new_name)
                if not os.path.isdir(new_path):
                    os.mkdir(new_path)

                print('change raw image full name:' + raw_full_name)
                print(' --->save new image full name:' + new_full_name)
                cmd_str = 'cp "' + raw_full_name + '" "' + new_full_name + '"'
                if sp.call(cmd_str, shell=True) != 0:
                    print('Fail to run ' + cmd_str)
                    sys.exit(-1)
                file_name_count += 1


def get_image_feature(image_list, feature_set):
    feature_extractor = df.DeepFeature()

    with open(image_list, 'r') as fid:
        for full_name in fid.readlines():
            full_name = full_name.strip('\n')
            print('process image feature:' + full_name)
            image_feature = feature_extractor.get_feature(full_name)
            save_file = generate_save_file(feature_set, full_name)
            save_image_feature(image_feature, save_file)


def process_image_feature(full_name):
    import deep_feature as df
    feature_extractor = df.DeepFeature()
    print('full_name:' + full_name)
    image_feature = feature_extractor.get_feature(full_name)
    save_file = generate_save_file(feature_set, full_name)
    #save_file = generate_save_file('./feature_set', full_name)
    save_image_feature(image_feature, save_file)


def save_image_feature(image_feature, file_name):
    feature_np = np.array(image_feature)

    with open(file_name, 'w') as fid:
        count = 0
        for datum in feature_np.flat:
            if count == 0:
                fid.write(str(datum))
            else:
                fid.write(',' + str(datum))
            count += 1


def generate_save_file(feature_set, full_name):
    pos = full_name[2:].find('/')
    sub_name = full_name[(2+pos):full_name.rfind('.')] + '.txt'
    save_name = feature_set + sub_name
    dir_name = os.path.dirname(save_name)

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    if not os.path.isfile(save_name):
        os.system('touch ' + save_name)
    return save_name


def test_algo_model(test_image_file_list, feature_folder, test_data_file):
    # step-1: build feature, label pair data
    print('step-1: build feature label pair data...')
    test_image_list = read_name_list_file(test_image_file_list)
    test_feature_file_list = get_feature_file(test_image_list, feature_folder)
    build_svm_data(test_feature_file_list, test_data_file)

    # step-2: predict image class
    print('step-2: predict image classs...')
    # os.system('./liblinear/predict -b 1 ' + test_data_file + ' trained_class_model' + ' predict_output_result')
    cmd_str = './liblinear/predict -b 1 ' + test_data_file + ' trained_class_model' + ' predict_output_result'
    if sp.call(cmd_str, shell=True) != 0:
        print('Fail to run ' + cmd_str)
        sys.exit(-1)


def get_feature_file(image_name_list, feature_folder):
    feature_name_list = []
    for image_name in image_name_list:
        image_name = image_name.strip('\n')
        pos = image_name[2:].find('/')
        sub_name = image_name[(2+pos):image_name.rfind('.')] + '.txt'
        save_name = feature_folder + sub_name
        feature_name_list.append(save_name)
    return feature_name_list


def read_name_list_file(file_name):
    out_list = []
    with open(file_name) as fid:
        for line in fid.readlines():
            line = line.strip('\n')
            out_list.append(line)
    return out_list


def build_svm_data(feature_file_list, svm_data):
    with open(svm_data, 'w+') as wfid:
        for file_name in feature_file_list:
            fields = file_name.split('/')
            class_name = fields[2]
            label = class_name[class_name.find('_')+1:]
            with open(file_name, 'r') as rfid:
                dense_feature = [float(s) for s in rfid.read().split(',')]
                sparse_feature = convert_dense_to_sparse(dense_feature)
                if len(sparse_feature) > 0 and len(label) > 0:
                    wfid.write(label + ' ' + sparse_feature + '\n')


def convert_dense_to_sparse(dense_list):
    sparse_list = ''
    for idx, val in enumerate(dense_list):
        if abs(val) > MY_ZERO_VALUE:
            # svm sparse feature index is start from 1.
            kv_str = str(idx+1) + ':' + str(val)
            if len(sparse_list) > 0:
                sparse_list += ' ' + kv_str
            else:
                sparse_list = kv_str
    return sparse_list


if __name__ == '__main__':
    start_time = time.time()
    image_class_real()
    print('all steps have finished!')
    stop_time = time.time()
    print('Total consume time:' + str(stop_time - start_time))
