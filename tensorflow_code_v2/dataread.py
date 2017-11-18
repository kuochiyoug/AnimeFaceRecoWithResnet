import tensorflow as tf
from IPython.terminal.debugger import set_trace as keyboard
from model.Resnet import Resnet

filename_queue = tf.train.string_input_producer(
tf.train.match_filenames_once("/home/baxter/dataset/animeface/animeface-character-dataset/thumb/*/*.png"))

image_reader = tf.WholeFileReader()
key, image_file = image_reader.read(filename_queue)
S = tf.string_split([key],'/')
length = tf.cast(S.dense_shape[1],tf.int32)
# adjust constant value corresponding to your paths if you face issues. It should work for above format.
label = S.values[length-tf.constant(2,dtype=tf.int32)]
#keyboard()
#label = tf.string_to_number(label,out_type=tf.int32)
image = tf.image.decode_png(image_file)

# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    #tf.initialize_all_variables().run()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    for i in xrange(6):
        # Get an image tensor and print its value.
        key_val,label_val,image_tensor = sess.run([key,label,image])
        print(image_tensor.shape)
        print(key_val)
        print(label_val)
        keyboard()


    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
