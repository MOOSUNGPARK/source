import tensorflow as tf
import Portpolio.Object_detection.Yolo_mobilenet.model.config as cfg


sess=tf.Session()

print(cfg.WEIGHTS_FILE)
restorer = tf.train.import_meta_graph(cfg.WEIGHTS_FILE + '.meta')
restorer.restore(sess, tf.train.latest_checkpoint('C:\\python\\source\\Portpolio\\Object_detection\\Yolo_mobilenet\\data\\train\\weights'))




# saver = tf.train.import_meta_graph('my_test_model-1000.meta')
# saver.restore(sess,tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
for op in tf.get_default_graph().get_operations():
    print(op.name)


