import tensorflow as tf
import glob as glob
import os


def get_cifar_subset(path="/storage/niranjan.rajesh_asp24/cifar10/cifar10/train/", class_name='dog'):
  batch_size = 32
  img_height = 32
  img_width = 32

  cifar_classes = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4,
                'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

  assert class_name in cifar_classes.keys(), "Invalid class name"
  print(f"Getting data for {class_name}")

  image_count = len(os.listdir(f'{path}/{class_name}')) 
  list_ds = tf.data.Dataset.list_files(f'{path}{class_name}/*.png', shuffle=False)
  list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
  print(f"Image count: {image_count}")
  print("Dataset Size: ", tf.data.experimental.cardinality(list_ds).numpy())

  

  val_size = int(image_count * 0.2)
  train_ds = list_ds.skip(val_size)
  val_ds = list_ds.take(val_size)

  def get_label():
    label = cifar_classes[class_name]
    return label

  def decode_img(img):
    print("Decoding image")
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])

  def process_path(file_path):
    print("Processing path")
    label = get_label()
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

  # for f in train_ds.take(5):
  #   print(f[0].numpy()[0])
  #   print(f[1])

  return train_ds, val_ds


  
