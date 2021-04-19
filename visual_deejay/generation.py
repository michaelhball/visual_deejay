import numpy as np

__all__ = ['get_noise_range']


def get_noise_range(out_size, generator_resolution):
    """ Gets the correct noise dimensions for a given resolution of StyleGAN2

    :param out_size: (int) output image size we want (can be different to generator e.g. doubling trick)
    :param generator_resolution: (int) output image resolution of the generator we are using
    :return: A range (min, max, & exponent) used to generate the sizes of the required noise vars.
    """

    log_max_res = int(np.log2(out_size))
    log_min_res = 2 + (log_max_res - int(np.log2(generator_resolution)))
    range_min = 2 * log_min_res + 1
    range_max = 2 * (log_max_res + 1)
    side_fn = lambda x: int(x / 2)
    return range_min, range_max, side_fn
