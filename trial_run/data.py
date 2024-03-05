import tensorflow as tf
import glob as glob
import os


def get_cifar_subset(path="/storage/niranjan.rajesh_asp24/cifar10/cifar10/train/", classes=['cat', 'dog']):
  batch_size = 32
  img_height = 32
  img_width = 32

  c1 = classes[0]
  c2 = classes[1]

  cifar_classes = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4,
                'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

  assert (c1 in cifar_classes.keys()) and (c2 in cifar_classes.keys()), "Invalid class names"
  print(f"Getting data for {c1} and {c2}")
  image_count = len(os.listdir(f'{path}/{c1}')) + len(os.listdir(f'{path}/{c2}'))
  list_ds = tf.data.Dataset.list_files(f'{path}{c1}/*.png', shuffle=False)
  list_ds2 = tf.data.Dataset.list_files(f'{path}{c2}/*.png', shuffle=False)
  list_ds = list_ds.concatenate(list_ds2)


  list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
  print(f"Image count: {image_count}")
  print("Concatenated Dataset Size: ", tf.data.experimental.cardinality(list_ds).numpy())

  for f in list_ds.take(5):
    print(f.numpy())

  val_size = int(image_count * 0.2)
  train_ds = list_ds.skip(val_size)
  val_ds = list_ds.take(val_size)

  def get_label(file_path):
    print("Getting labels")
    # Convert the path of image to class label
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    label = parts[-2] == classes
    # Integer encode the label
    return tf.argmax(label)

  def decode_img(img):
    print("Decoding image")
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])

  def process_path(file_path):
    print("Processing path")
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

  train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
  val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

  def configure_for_performance(ds):
    print("Configuring for performance")
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

  train_ds = configure_for_performance(train_ds)
  val_ds = configure_for_performance(val_ds)

  return train_ds, val_ds


  
