"""
Helper StyleGAN functions (for playing with a trained network, not for training)

NB: Requires stylegan2 repository to be in the same directory as where the code importing these functions was run
"""

import numpy as np
import pickle
import PIL.Image
import sys

from tqdm.notebook import tqdm

sys.path.append("./stylegan2")
import dnnlib
import dnnlib.tflib as tflib

__all__ = ['load_networks', 'convert_z_to_w', 'generate_images_from_ws', 'generate_images_from_zs',
           'generate_images_from_seeds', 'interpolate_linear']


def load_networks(file_path):
    """

    :param file_path:
    :return:
    """

    stream = open(file_path, 'rb')
    tflib.init_tf()
    with stream:
        G, D, Gs = pickle.load(stream, encoding='latin1')
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    return G, D, Gs, noise_vars
^

def convert_z_to_w(Gs, latent, truncation_psi=0.7, truncation_cutoff=9):
    """ Converts from

    :param Gs: TF StyleGAN2 generator network
    :param latent: latent vector z we want to convert
    :param truncation_psi:
    :param truncation_cutoff:
    :return: w vector
    """

    dlatent = Gs.components.mapping.run(latent, None)  # [seed, layer, component]
    dlatent_avg = Gs.get_var('dlatent_avg')  # [component]
    for i in range(truncation_cutoff):
        dlatent[0][i] = (dlatent[0][i] - dlatent_avg) * truncation_psi + dlatent_avg
    return dlatent


def generate_images_from_ws(Gs, dlatents, truncation_psi):
    """

    :param Gs:
    :param dlatents:
    :param truncation_psi:
    :return:
    """

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = truncation_psi
    # dlatent_avg = Gs.get_var('dlatent_avg')  # [component]

    imgs = []
    for row, dlatent in tqdm(enumerate(dlatents), desc="Generating images"):
        # row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(truncation_psi, [-1, 1, 1]) + dlatent_avg
        # dl = (dlatent - dlatent_avg) * truncation_psi + dlatent_avg
        row_images = Gs.components.synthesis.run(dlatent,  **Gs_kwargs)
        imgs.append(PIL.Image.fromarray(row_images[0], 'RGB'))
    return imgs


def generate_images_from_zs(Gs, noise_vars, zs, truncation_psi):
    """

    :param Gs:
    :param noise_vars:
    :param zs:
    :param truncation_psi:
    :return:
    """

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if not isinstance(truncation_psi, list):
        truncation_psi = [truncation_psi] * len(zs)

    imgs = []
    for z_idx, z in tqdm(enumerate(zs), total=len(zs), desc="Generating images"):
        Gs_kwargs.truncation_psi = truncation_psi[z_idx]
        noise_rnd = np.random.RandomState(1)
        tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
        images = Gs.run(z, None, **Gs_kwargs)  # [mini_batch, height, width, channel]
        imgs.append(PIL.Image.fromarray(images[0], 'RGB'))
    return imgs


def generate_zs_from_seeds(Gs, seeds):
    """

    :param Gs:
    :param seeds:
    :return:
    """

    zs = []
    for seed_idx, seed in enumerate(seeds):
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:])  # [minibatch, component]
        zs.append(z)
    return zs


def generate_images_from_seeds(Gs, seeds, truncation_psi, noise_vars=None):
    """

    :param Gs:
    :param seeds:
    :param truncation_psi:
    :param noise_vars:
    :return:
    """

    if noise_vars is None:
        noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    return generate_images_from_zs(Gs, noise_vars, generate_zs_from_seeds(Gs, seeds), truncation_psi)


def interpolate_linear(zs, steps):
    """ Function to perform linear interpolation between vectors

    :param zs: iterable of z variables
    :param steps: number of steps that should should be generated between each latent (=> # frames)
    :return:
    """

    out = []
    for i in range(len(zs) - 1):
        for index in range(steps):
            fraction = index / float(steps)
            out.append(zs[i + 1] * fraction + zs[i] * (1 - fraction))
    return out
