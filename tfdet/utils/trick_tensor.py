import tensorflow.compat.v1 as tf
from tfdet.utils.shape_utils import shape_list
def matmul_gather_on_zeroth_axis(params, indices, scope=None):
  """Matrix multiplication based implementation of tf.gather on zeroth axis.
  TODO(rathodv, jonathanhuang): enable sparse matmul option.
  Args:
    params: A float32 Tensor. The tensor from which to gather values.
      Must be at least rank 1.
    indices: A Tensor. Must be one of the following types: int32, int64.
      Must be in range [0, params.shape[0])
    scope: A name for the operation (optional).
  Returns:
    A Tensor. Has the same type as params. Values from params gathered
    from indices given by indices, with shape indices.shape + params.shape[1:].
  """
  with tf.name_scope(scope, 'MatMulGather'):
    params_shape =shape_list(params)
    indices_shape = shape_list(indices)
    params2d = tf.reshape(params, [params_shape[0], -1])
    indicator_matrix = tf.one_hot(indices, params_shape[0])
    gathered_result_flattened = tf.matmul(indicator_matrix, params2d)
    return tf.reshape(gathered_result_flattened,
                      tf.stack(indices_shape + params_shape[1:]))

def gather_based_on_match(
    input_tensor, 
    unmatched_value,
    ignored_value,
    
    param_to_gather,
    name_ops='gather_normal'
):
    input_tensor = tf.concat(
        [tf.stack([ignored_value, unmatched_value]),
            input_tensor],
        axis=0)
    gather_indices = tf.maximum(param_to_gather + 2, 0)
    if name_ops =='gather_matmul':
        gathered_tensor = matmul_gather_on_zeroth_axis(input_tensor, gather_indices)
    else:
        gathered_tensor = tf.gather(input_tensor, gather_indices)
    return gathered_tensor
