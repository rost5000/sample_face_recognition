from PIL import Image
import argparse
from pathlib import Path
from src.python.insight_face.config import get_config
from mtcnn import MTCNN
from src.python.insight_face.Learner import face_learner
from src.python.insight_face.utils import load_facebank, prepare_facebank
import numpy as np
import os
import time


# Результат прогназа сетки:
# predict - предположительный класс, к которому отн-ся фото
# dist - евклидово расстояние до предположительного класса
# соотв-но в самой сетке выбор класса происходит по минимальному расстоянию
# threshold - параметр-расстояние, который мы можем задать
# actual - действительный класс изображения

def get_tp_fp_tn_fn(dist: float, actual,
                    predict, path_to_file: Path):
    tp, fp, tn, fn = 0, 0, 0, 0
    # threshold = 0.8, actual = 103, dist = 0.6024671792984009, predict = 103
    # threshold = 0.8, actual = 103, dist = 0.8871978521347046, predict = None
    predict_is_same = predict != None  # predict != 'None' => 1, predict == 'None' => 0
    actual_is_same = actual != None  # actual != None => 1, actual == None => 0

    # threshold,actual,dist,predict,path'
    if ((predict == None) and (actual != None)) or ((predict != None) and (actual == None)):
        with open('mistakes.csv', 'a') as file:
            file.write(f'{learner.threshold},{actual},{dist},{predict},{path_to_file}\n')
    # if not actual_issame:
    #     print(f'threshold={threshold}, dist={dist}, actual_person={actual},'
    #           f' predict_person={predict}, path_to_file={path_to_file}')

    if predict_is_same and actual_is_same:  # расстояние до класса мало и фото одинаковые
        tp += 1
    elif predict_is_same and not actual_is_same:  # расстояние до класса мало, хотя фото разные
        fp += 1
    elif not predict_is_same and not actual_is_same:  # расстояние до класса большое и действ-но фото разные
        tn += 1
    elif not predict_is_same and actual_is_same:  # расстояние до класса большое, хотя фото одинаковые
        fn += 1
    else:
        print('Неправильно задано условие')
    return tp, fp, tn, fn


def get_face(file_path, path_del_files):
    if path_del_files.name != '.DS_Store':
        try:
            img = Image.open(file_path)
        except Exception:
            print(f"\nRemove file {file_path}\n")
            with open(path_del_files, 'a') as f:
                f.write(f'{file_path},cant_open\n')
                os.remove(file_path)
        if img.size != (112, 112):
            face_to_ret = mtcnn.align(img)
            if not face_to_ret:
                print(f"\nRemove file {file_path}\n")
                with open(path_del_files, 'a') as f:
                    f.write(f'{file_path},mtccn_err\n')
                os.remove(file_path)
                return None
            return face_to_ret
    else:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save", action="store_true")
    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1., type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")
    args = parser.parse_args()

    conf = get_config(False)

    mtcnn = MTCNN()
    print('mtcnn loaded')
    with open('mistakes.csv', 'w') as file:
        file.write('threshold,actual,dist,predict,path\n')
    learner = face_learner(conf)
    learner.threshold = 0.8
    learner.load_state(conf, 'cpu_final.pth', True, True)
    learner.model.eval()

    if False:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta=False)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)

    print('databank is ready')
    tp_all, fp_all, tn_all, fn_all = 0, 0, 0, 0
    paths_test = list((Path(conf.data_path) / 'test').iterdir())
    counter = 1
    len_paths = len(paths_test)
    path_del_files = conf.data_path / 'deleted_files.txt'

    # create True faces:
    faces_true = []
    faces_false = []
    test_y_true = []
    test_y_false = []
    path_file_true = []
    path_file_false = []
    for path in paths_test:
        if not path.is_file():
            actual_person = str(path).split('/')[-1]  # Path.name()
            for file in path.iterdir():
                if file.is_file():
                    face = get_face(file, path_del_files)
                    if not face:
                        continue
                    if actual_person != '0':
                        faces_true.append(face)
                        test_y_true.append(actual_person)
                        path_file_true.append(file)
                    else:
                        faces_false.append(face)
                        test_y_false.append(None)
                        path_file_false.append(file)
                else:
                    continue
        else:
            continue

    test_y_predict_true = []
    test_y_predict_false = []
    for indx in range(len(faces_true)):
        print(f'{counter}/{len(faces_true)} verify data')
        counter += 1
        begin = time.time()
        results, dists = learner.infer(conf, [faces_true[indx]], targets, args.tta)
        end = time.time()
        print(f'time={end - begin}')

        predict_temp = names[results[0] + 1] if results[0] + 1 != 0 else None
        test_y_predict_true.append(predict_temp)
        tp_temp, fp_temp, tn_temp, fn_temp = get_tp_fp_tn_fn(
            dist=dists[0], actual=test_y_true[indx],
            predict=predict_temp, path_to_file=path_file_true[indx])
        tp_all += tp_temp
        fp_all += fp_temp
        tn_all += tn_temp
        fn_all += fn_temp

    counter = 0
    for indx in range(len(faces_false)):
        print(f'{counter}/{len(faces_false)} verify data')
        counter += 1
        results, dists = learner.infer(conf, [faces_false[indx]], targets, args.tta)
        predict_temp = names[results[0] + 1] if results[0] + 1 != 0 else None
        test_y_predict_false.append(predict_temp)
        tp_temp, fp_temp, tn_temp, fn_temp = get_tp_fp_tn_fn(
            dist=dists[0], actual=test_y_false[indx],
            predict=predict_temp, path_to_file=path_file_false[indx])
        tp_all += tp_temp
        fp_all += fp_temp
        tn_all += tn_temp
        fn_all += fn_temp

    true_positive = np.mean(
        [test_y_true[indx] == test_y_predict_true[indx] for indx in range(len(test_y_predict_true))])
    false_negative = np.mean([test_y_predict_true[indx] == None for indx in range(len(test_y_predict_true))])
    true_negative = np.mean(
        [test_y_false[indx] == test_y_predict_false[indx] for indx in range(len(test_y_predict_false))])
    false_positive = np.mean([test_y_predict_false[indx] != None for indx in range(len(test_y_predict_false))])

    tpr, fpr, tnr, fnr = 0, 0, 0, 0
    tpr = tp_all / (tp_all + fn_all) if (tp_all + fn_all) != 0 else 0
    fpr = fp_all / (fp_all + tn_all) if (fp_all + tn_all) != 0 else 0
    tnr = tn_all / (fp_all + tn_all) if (fp_all + tn_all) != 0 else 0
    fnr = fn_all / (tp_all + fn_all) if (tp_all + fn_all) != 0 else 0
    _sum = tp_all + fn_all + fp_all + tn_all
    tpr_to_sum = tp_all / _sum
    fpr_to_sum = fp_all / _sum
    tnr_to_sum = tn_all / _sum
    fnr_to_sum = fn_all / _sum
    accuracy = (tp_all + tn_all) / _sum  # 0.93

    print(f'faces_true={faces_true}')
    print(f'faces_false={faces_false}')
    print(f'test_y_true={test_y_true}')
    print(f'test_y_false={test_y_false}')
    print(f'test_y_predict_true={test_y_predict_true}')
    print(f'test_y_predict_false={test_y_predict_false}')

    print(f"TP={true_positive}")
    print(f"FP={false_positive}")
    print(f"TN={true_negative}")
    print(f"FN={false_negative}")

    print(f'tp_all    = {tp_all}')
    print(f'fp_all    = {fp_all}')
    print(f'tn_all    = {tn_all}')
    print(f'fn_all    = {fn_all}')

    print(f'threshold = {learner.threshold}')
    print(f'tpr    = {tpr}')
    print(f'fpr    = {fpr}')
    print(f'tnr    = {tnr}')
    print(f'fnr    = {fnr}')
    print(f'tpr_to_sum    = {tpr_to_sum}')
    print(f'fpr_to_sum    = {fpr_to_sum}')
    print(f'tnr_to_sum    = {tnr_to_sum}')
    print(f'fnr_to_sum    = {fnr_to_sum}')
    print(f'accuracy = {accuracy}')
