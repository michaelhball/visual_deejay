import imageio
import IPython.display
import math
import numpy as np
import os
import PIL.Image

from base64 import b64encode
from io import BytesIO
from IPython.display import HTML
from tqdm.notebook import tqdm

__all__ = ['show_image', 'show_video', 'save_images', 'create_image_grid', 'create_video_from_images']


def show_image(image, image_format='png', jpeg_fallback=True):
    """ Displays a single image in Jupyter/Colab

    :param image: some array format image
    :param image_format: (str) image format
    :param jpeg_fallback:
    :return: IPython display object (displaying image in notebook)
    """

    image = np.asarray(image, dtype=np.uint8)
    str_file = BytesIO()
    PIL.Image.fromarray(image).save(str_file, image_format)
    im_data = str_file.getvalue()
    disp = IPython.display.display(IPython.display.Image(im_data))
    return disp


def show_video(video_file_name=None, video_obj=None):
    """ Displays an interactive video in notebook.

    :param video_file_name: (path) path to video file to load
    :param video_obj: (??) video file to display
    :return: None
    """

    mp4 = open(video_file_name, 'rb').read()
    data_uri = "data:video/mp4;base64," + b64encode(mp4).decode()
    HTML("""
      <video width=400 controls>
        <source src="%s" type="video/mp4">
      </video>
    """ % data_uri)


def save_images(images, save_dir):
    """ Writes a sequence of images to a specified directory

    :param images: sequence of image objects to save
    :param save_dir: (path) path to directory in which we want to save images
    :return: success indicator
    """

    try:
        for idx, img in tqdm(enumerate(images), total=len(images), desc="Saving images"):
            file = os.path.join(save_dir, f'{idx}.png')
            img.save(file)
        return True
    except:
        return False


def create_image_grid(images, scale=0.25, rows=1):
    """ Create grid of images (e.g. for plotting style mixing | interpolation).

    :param images: iterable of images to plot (~assumes they are same size)
    :param scale: (float) how much images should be scaled
    :param rows: (int) number of rows grid should have (=> columns = automatic)
    :return: PIL image containing image grid
    """

    w, h = images[0].size
    w, h = int(w * scale), int(h * scale)
    height = rows * h
    cols = math.ceil(len(images) / rows)
    width = cols * w
    canvas = PIL.Image.new('RGBA', (width, height), 'white')
    for i, img in enumerate(images):
        img = img.resize((w, h), PIL.Image.ANTIALIAS)
        canvas.paste(img, (w * (i % cols), h * (i // cols)))
    return canvas


def create_video_from_images(images, video_file_name):
    """ Creates a video from a sequence of images

    :param images: sequence of images to convert --> video
    :param video_file_name: (path) name of output video file
    :return: success indicator
    """

    try:
        with imageio.get_writer(video_file_name, mode='I') as writer:
            for image in tqdm(images, desc="creating video"):
                writer.append_data(np.array(image))
        return False
    except:
        return False
