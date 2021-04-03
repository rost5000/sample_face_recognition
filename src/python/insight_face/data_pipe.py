from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def de_preprocess(tensor):
    return tensor * 0.5 + 0.5
