import tensorflow as tf

def convert_tf1_to_tf2(checkpoint_path, output_prefix):
    """Converts a TF1 checkpoint to TF2.

    To load the converted checkpoint, you must build a dictionary that maps
    variable names to variable objects.
    ```
    ckpt = tf.train.Checkpoint(vars={name: variable})
    ckpt.restore(converted_ckpt_path)

    ```

    Args:
    checkpoint_path: Path to the TF1 checkpoint.
    output_prefix: Path prefix to the converted checkpoint.

    Returns:
    Path to the converted checkpoint.
    """
    vars = {}
    reader = tf.train.load_checkpoint(checkpoint_path)
    dtypes = reader.get_variable_to_dtype_map()
    for key in dtypes.keys():
        vars[key] = tf.Variable(reader.get_tensor(key))
    return tf.train.Checkpoint(vars=vars).save(output_prefix)



def print_checkpoint(save_path):
  reader = tf.train.load_checkpoint(save_path)
  shapes = reader.get_variable_to_shape_map()
  dtypes = reader.get_variable_to_dtype_map()
  print(f"Checkpoint at '{save_path}':")
  for key in shapes:
    print(f"  (key='{key}', shape={shapes[key]}, dtype={dtypes[key].name}") #, "f"value={reader.get_tensor(key)})")


# convert_tf1_to_tf2('/home/yael/PycharmProjects/PatchDist/chk', '/home/yael/PycharmProjects/PatchDist/chk_tf2')
print_checkpoint('/home/yael/PycharmProjects/PatchDist/chk')