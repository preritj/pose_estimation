import tensorflow as tf


def combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.
    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.
    Args:
      tensor: A tensor of any type.
    Returns:
      A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_tensor_shape = tensor.shape.as_list()
    dynamic_tensor_shape = tf.shape(tensor)
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape


def upsample(x):
    shape = x.get_shape().as_list()
    r1 = tf.reshape(x, [shape[0], shape[1] * shape[2], 1, shape[3]])
    r1_l = tf.pad(r1, [[0, 0], [0, 0], [0, 1], [0, 0]])
    r1_r = tf.pad(r1, [[0, 0], [0, 0], [1, 0], [0, 0]])
    r2 = tf.add(r1_l, r1_r)
    r3 = tf.reshape(r2, [shape[0], shape[1], shape[2] * 2, shape[3]])
    r3_l = tf.pad(r3, [[0, 0], [0, 0], [0, shape[2] * 2], [0, 0]])
    r3_r = tf.pad(r3, [[0, 0], [0, 0], [shape[2] * 2, 0], [0, 0]])
    r4 = tf.add(r3_l, r3_r)
    r5 = tf.reshape(r4, [shape[0], shape[1] * 2, shape[2] * 2, shape[3]])
    return r5


def nearest_neighbor_upsampling(input_tensor, scale):
    """Nearest neighbor upsampling implementation.
    Nearest neighbor upsampling function that maps input tensor with shape
    [batch_size, height, width, channels] to [batch_size, height * scale
    , width * scale, channels]. This implementation only uses reshape and tile to
    make it compatible with certain hardware.
    Args:
      input_tensor: A float32 tensor of size [batch, height_in, width_in,
        channels].
      scale: An integer multiple to scale resolution of input data.
    Returns:
      data_up: A float32 tensor of size
        [batch, height_in*scale, width_in*scale, channels].
    """
    shape = combined_static_and_dynamic_shape(input_tensor)
    shape_before_tile = [shape[0], shape[1], 1, shape[2], 1, shape[3]]
    shape_after_tile = [shape[0], shape[1] * scale, shape[2] * scale, shape[3]]
    data_reshaped = tf.reshape(input_tensor, shape_before_tile)
    resized_tensor = tf.tile(data_reshaped, [1, 1, scale, 1, scale, 1])
    resized_tensor = tf.reshape(resized_tensor, shape_after_tile)
    return resized_tensor


def non_max_suppression(input_tensor, window_size, name='nms'):
    # input: B x H x W x C
    pooled = tf.nn.max_pool(input_tensor,
                            ksize=[1, window_size, window_size, 1],
                            strides=[1, 1, 1, 1], padding='SAME')
    output = tf.where(tf.equal(input_tensor, pooled),
                      input_tensor,
                      tf.zeros_like(input_tensor),
                      name=name)
    # output: B X W X H x C
    return output
