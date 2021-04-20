import numpy as np
from PIL import Image
from torchvision import transforms as trans
from src.python.insight_face.data_pipe import de_preprocess
import torch
from src.python.insight_face.model import l2_norm
import cv2
import os


def prepare_facebank(conf, model, mtcnn, tta=True):
    model.eval()
    embeddings = []
    names = ['Unknown']
    path_del_files = conf.data_path / 'deleted_files.txt'
    paths = list((conf.data_path / 'train').iterdir())
    counter = 1
    len_paths = len(paths)
    for path in paths:
        print(f'{counter}/{len_paths} preparing data')
        counter += 1
        if path.is_file():
            continue
        embs = []
        for file in path.iterdir():
            if not file.is_file():
                continue
            try:
                img = Image.open(file)
            except:
                print(f"\nRemove file {file}\n")
                with open(path_del_files, 'a') as f:
                    f.write(f'{file},cant_open\n')
                    os.remove(file)
            if img.size != (112, 112):
                img = mtcnn.align(img)
                if img == False:
                    print(f"\nRemove file {file}\n")
                    with open(path_del_files, 'a') as f:
                        f.write(f'{file},mtccn_err\n')
                    os.remove(file)
                    continue
            with torch.no_grad():
                if tta:
                    mirror = trans.functional.hflip(img)
                    emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                    emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                    embs.append(l2_norm(emb + emb_mirror))
                else:
                    embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        if len(embs) == 0:
            print(f"\nRemove file {file}\n")
            with open(path_del_files, 'a') as f:
                f.write(f'{file},no_embs\n')
            os.remove(file)
            continue
        embedding = torch.cat(embs).mean(0, keepdim=True)
        embeddings.append(embedding)
        names.append(path.name)
    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, conf.facebank_path / 'facebank.pth')
    np.save(conf.facebank_path / 'names', names)
    return embeddings, names


def get_face(mtcnn, frame):
    img_temp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_temp)
    img = mtcnn.align(img)
    return img if img else None


# def get_emb(mtcnn, frame):
#     img_temp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(img_temp)
#     img = mtcnn.align(img)
#     if img:
#         return img

def get_emb_from_frame(conf, model, mtcnn, frame, tta=True):
    img_temp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_temp)
    detected_img = mtcnn.align(img)
    if detected_img:
        with torch.no_grad():
            if tta:
                mirror = trans.functional.hflip(detected_img)
                emb = model(conf.test_transform(detected_img).to(conf.device).unsqueeze(0))
                emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                return l2_norm(emb + emb_mirror)
            else:
                return model(conf.test_transform(detected_img).to(conf.device).unsqueeze(0))


def load_facebank(conf):
    embeddings = torch.load(conf.facebank_path / 'facebank.pth')
    names = np.load(conf.facebank_path / 'names.npy')
    return embeddings, names


def face_reader(conf, conn, flag, boxes_arr, result_arr, learner, mtcnn, targets, tta):
    while True:
        try:
            image = conn.recv()
        except:
            continue
        try:
            bboxes, faces = mtcnn.align_multi(image, limit=conf.face_limit)
        except:
            bboxes = []

        results = learner.infer(conf, faces, targets, tta)

        if len(bboxes) > 0:
            print('bboxes in reader : {}'.format(bboxes))
            bboxes = bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1, -1, 1, 1]  # personal choice
            assert bboxes.shape[0] == results.shape[0], 'bbox and faces number not same'
            bboxes = bboxes.reshape([-1])
            for i in range(len(boxes_arr)):
                if i < len(bboxes):
                    boxes_arr[i] = bboxes[i]
                else:
                    boxes_arr[i] = 0
            for i in range(len(result_arr)):
                if i < len(results):
                    result_arr[i] = results[i]
                else:
                    result_arr[i] = -1
        else:
            for i in range(len(boxes_arr)):
                boxes_arr[i] = 0  # by default,it's all 0
            for i in range(len(result_arr)):
                result_arr[i] = -1  # by default,it's all -1
        print('boxes_arr ： {}'.format(boxes_arr[:4]))
        print('result_arr ： {}'.format(result_arr[:4]))
        flag.value = 0


hflip = trans.Compose([
    de_preprocess,
    trans.ToPILImage(),
    trans.functional.hflip,
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs
