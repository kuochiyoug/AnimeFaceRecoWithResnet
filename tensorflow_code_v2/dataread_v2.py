import tensorflow as tf
from IPython.terminal.debugger import set_trace as keyboard
import os, glob
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from PIL import Image
from model.Resnet import ResNet



def read_labeled_image_list_file(image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(int(label))
    return filenames, labels

def read_labeled_image_list_from_folder(data_path):
    """Reads a folder containing pathes and labeles
    Args:
       image_list: a list with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    #Data path setup
    folders = sorted(os.listdir(data_path))
    cats = []#Categorys list
    labels_id = []
    img_filelist = []
    for folder in folders:
        if os.path.isfile(data_path + folder + "/" + "ignore"):
            print("Folder "+ folder + "is ignored!")
            continue
        else:
            cats.append(folder)
            label_id = cats.index(folder)
            img_filelist_in_folder = glob.glob(data_path + folder + "/"+"*.png")
            for imgfile in img_filelist_in_folder:
                #img = Image.open(imgfile)
                #img = np.asarray(np.float32(img))
                #if img.shape[2] != 3:
                #    continue
                #all_data.append([imgfile,label])
                img_filelist.append(imgfile)
                labels_id.append(label_id)
    print("labels="+str(len(cats)))
    return img_filelist, labels_id ,cats

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)
    return example, label

def preprocess_image(image_tensor):
    """
    The preprocessor for per image tensor
    """
    image_tensor = tf.image.resize_images(image_tensor,(224,224))
    #image_tensor = tf.transpose(image_tensor,perm=[2,0,1])
    #img = img.transpose(2,0,1)
    return image_tensor

def preprocess_label(label_id_list,cat_list):
    """
    ww
    """
    label_list = []
    for label_id in label_id_list:
        label = np.zeros(len(cat_list),np.int32)
        label[label_id]=1
        label_list.append(label)
    label_list = np.vstack(label_list)
    return label_list



folder="/home/baxter/dataset/animeface/animeface-character-dataset/thumb/"
num_epochs=50
seed=1234
batch_size=50

# Reads pfathes of images together with their labels
#image_list, label_list = read_labeled_image_list(filename)
image_list, label_id_list, cat_list= read_labeled_image_list_from_folder(folder)

label_list = preprocess_label(label_id_list,cat_list)

images = ops.convert_to_tensor(image_list, dtype=tf.string)
#labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
#images = tf.convert_to_tensor(image_list, dtype=tf.string)
#labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
#cats = tf.convert_to_tensor(cat_list,dtype=tf.string)


# Makes an input queue
input_queue = tf.train.slice_input_producer([images, label_list],
                                            num_epochs=num_epochs,
                                            shuffle=True,
                                            seed=seed)
image, label = read_images_from_disk(input_queue)



# Optional Preprocessing or Data Augmentation
# tf.image implements most of the standard image augmentation
image = preprocess_image(image)



# Optional Image and Label Batching
image_batch, label_batch = tf.train.batch([image, label],
                                          batch_size=batch_size)
model = ResNet(name='ResNet',in_channels=3,seed=seed)
model.train = True

cfg = dict({
            'allow_soft_placement': False,
            'log_device_placement': True,
            'gpu_options': tf.GPUOptions()
        })
config = tf.ConfigProto(**cfg)
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
#keyborad()
with tf.device('/gpu:0'):
    pred_batch = model(image_batch)
    loss_tf = tf.losses.softmax_cross_entropy(label_batch,pred_batch)
    #accuracy_tf = tf.metrics.accuracy(label,pred_batch)
    #accuracy = tf.metr
    op = tf.train.AdamOptimizer()
    gradient = op.compute_gradients(loss_tf)
    train_op = op.apply_gradients(gradient)



with tf.Session(config = config) as sess:
    with tf.device('/device:gpu:0'):
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(10000):
            _,loss= sess.run([train_op,loss_tf])
            print i, loss
            #imgplot = plt.imshow(final_image[0])

        coord.request_stop()
        coord.join(threads)
