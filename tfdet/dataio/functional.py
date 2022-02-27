import tensorflow as tf 


def normalize(img, mean, std, max_pixel_value=255.0):
    mean = mean * max_pixel_value
    std = std * max_pixel_value
    denominator = tf.math.divide_no_nan(1.,std)
    img = img - mean
    img = img * denominator
    return img