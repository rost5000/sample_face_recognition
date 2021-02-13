import os, shutil, csv
from sklearn.model_selection import KFold

src_path = '../dataset/img_align_celeba'
dest_path = '../dataset/img_align_sorted'
dest_path_light = '../dataset/img_align_sorted_light'


hight_board = 1510
med_val_test = 605
med_val_train = 588
counter = 0
counter_test = 0
counter_train = 0
meta_info = []

kf = KFold(n_splits=2)
test_y, test_x, train_x, train_y = [[], [], [], []]

if __name__ == '__main__':
    for clss in os.listdir(dest_path):
        if hight_board < counter_test and hight_board < counter_train:
            break
        if not os.path.exists(dest_path_light + '/' + clss):
            os.mkdir(dest_path_light + '/' + clss)
        print("class: " + clss, end='\r')
        dates = os.listdir(dest_path + '/' + clss)
        train_index, test_index = kf.split(dates)
        for indx in  train_index[0]:
            train_y.append(clss)
            meta_info.append({'filename': dates[indx], 'is_train': 1, 'is_test': 0, 'count': counter_train, 'binary_class': 0 if counter_train > med_val_train else 1})
            counter_train = counter_train + 1
            shutil.copy(dest_path + '/' + clss + '/' + dates[indx], dest_path_light + '/' + clss + '/' + dates[indx])
        for indx in test_index[0]:
            test_y.append(clss)
            meta_info.append({'filename': dates[indx], 'is_train': 0, 'is_test': 1, 'count': counter_test, 'binary_class': 0 if counter_test > med_val_test else 1})
            counter_test = counter_test + 1
            shutil.copy(dest_path + '/' + clss + '/' + dates[indx], dest_path_light + '/' + clss + '/' + dates[indx])


    csv_columns = ['filename', 'is_train', 'is_test', 'count', 'binary_class']
    with open(dest_path_light + '/' + 'META_INFO.csv', 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()
        for inf in meta_info:
            writer.writerow(inf)
        csv_file.close()