from PIL import Image


def open_rgb_image(path, image_size):
    image = Image.open(path).convert("RGB")
    target_size = (image_size, image_size)
    if image.size != target_size:
        image = image.resize(target_size, Image.BICUBIC)
    return image
