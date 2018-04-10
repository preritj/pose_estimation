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
