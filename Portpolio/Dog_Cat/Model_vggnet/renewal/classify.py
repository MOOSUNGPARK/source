import tensorflow as tf
import numpy as np
from Portpolio.Dog_Cat.Model_vggnet.renewal.util import read_data
import Portpolio.Dog_Cat.Model_vggnet.renewal.config as cfg
from Portpolio.Dog_Cat.Model_vggnet.renewal.vggnet import Vggnet


model = Vggnet('Vggnet', cfg.LABEL_CNT)

def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + '/Assign') for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    print(assign_ops)
    print(feed_dict)
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())

# self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
# print('Restoring weigths from : ' + cfg.CKPT_DIR_PATH)
# self.saver.restore(self.sess, tf.train.latest_checkpoint(cfg.CKPT_DIR_PATH))


# First let's load meta graph and restore weights



    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    saver.restore(sess, tf.train.latest_checkpoint(cfg.CKPT_DIR_PATH))
    # saver = tf.train.import_meta_graph(cfg.CKPT_FILE + '-2385' + '.meta')
    # saver.restore(sess, cfg.CKPT_FILE + '-2385')

    graph = tf.get_default_graph()
    # graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # print([n.name for n in tf.get_default_graph().as_graph_def().node])
    for n in tf.get_default_graph().as_graph_def().node:
        print(n.name)
    #
    # # X = graph.get_tensor_by_name('Vggnet/initialize_scope/X_data:0')
    # # Y = graph.get_tensor_by_name('Vggnet/initialize_scope/Y_data:0')
    # # training = graph.get_tensor_by_name('Vggnet/initialize_scope/training:0')
    #
    # model_params = get_model_params()
    # # print(model_params)
    # restore_model_params(model_params)

    model.X = graph.get_tensor_by_name('Vggnet/initialize_scope/X_data:0')
    model.Y = graph.get_tensor_by_name('Vggnet/initialize_scope/Y_data:0')
    model.training = graph.get_tensor_by_name('Vggnet/initialize_scope/training:0')

    test_accuracy = []
    for idx in range(0, len(cfg.TEST_FILE_LIST), 2):
        test_x, test_y, test_total_size = read_data(cfg.TEST_FILE_LIST[idx], cfg.TEST_FILE_LIST[idx + 1])

        for start_idx in range(0, test_total_size, cfg.BATCH_SIZE):
            test_x_batch = test_x[start_idx:start_idx + cfg.BATCH_SIZE]
            test_y_batch = test_y[start_idx:start_idx + cfg.BATCH_SIZE]

            feed_dict = {model.X: test_x_batch, model.Y: test_y_batch, model.training: False}

            acc = sess.run([model.accuracy], feed_dict=feed_dict)
            test_accuracy.append(acc)

    print('Test Accuracy :', np.mean(np.array(test_accuracy)))
    print('Test Finished!')
    print('global_step :', sess.run(model.global_step))


