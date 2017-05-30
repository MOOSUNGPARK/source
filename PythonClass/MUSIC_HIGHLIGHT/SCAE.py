from __future__ import division
from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
from Data import read_data_sets


batch_size = 128

class StackedCAE(object):

  _weights_str = 'weights{0}'
  _biases_str  = 'biases{0}'
  _deconv_weights_str = 'weights{0}_deconv'
  _deconv_biases_str  = 'biases{0}_deconv'
 

  def __init__(self,shape, sess,test=False):
  

    self.__shape = shape # [(input_ch, input_dim),(first # filter, first filter size, pooling size), (second #filter, second filter size, second pooling size), (.. .. ..),(previous_units=640,hidden_size = 300),(previous_units=300,output_size=2)
    self.__num_conv_layers = len(self.__shape) - 3
    self.__test = test
    self.__num_feed_layers = 2
    self.__variables = {}
    self.__sess = sess
    self.__poolsize = {}
    self._setup_variables()

  def shape(self):
    return self.__shape

  def num_conv_layers(self):
    return self.__num_conv_layers

  def session(self):
    return self.__sess

  def __getitem__(self,item):
    return self.__variables[item]

  def __setitem__(self,key,value):
    self.__variables[key] = value

  def _setup_variables(self):
    (_,input_dim) = self.__shape[0]
    conv_str = 'conv{0}'
    with tf.variable_scope('SCAE_variables'):
      for i in xrange(self.__num_conv_layers):
        # Train Weights for pre-training
        name_w = self._weights_str.format(i+1)
        name_b = self._biases_str.format(i+1)
        name_deconv_w = self._deconv_weights_str.format(i+1)
        name_deconv_b = self._deconv_biases_str.format(i+1)
        scope_conv = conv_str.format(i+1)
        ## current layer's size
        (filter_num,filter_size,pool_size) = self.__shape[i+1]
        ## store pooling size of each layer
        self.__poolsize[i+1] = pool_size

        ## previous layer's size
        pre_shape = self.__shape[i]
        ## current conv layer's filter shape
        shape = [filter_size, filter_size, pre_shape[0], filter_num]
        b_shape = [filter_num]
        deconv_b_shape = [pre_shape[0]]


        # scope for each convolutional neural net
        with tf.variable_scope(scope_conv) as scope:

          w_init = tf.truncated_normal(shape,stddev=0.1)
          # w_init = tf.ones(shape)
          ## if first conv layer, initilaize filter by muliplying SRM feature
          if i==0:
            SRM = [[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,0,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]]
            SRM = np.asarray(SRM).astype('float32')/12
            if self.__test:
              SRM = [[1,2],[-1,-2]]
              SRM = np.asarray(SRM).astype('float32')
            SRM = np.tile(SRM,[pre_shape[0],filter_num,1,1])
            SRM = np.transpose(SRM,[2,3,0,1])
            SRM = tf.constant(SRM,name='SRM')
            w_init = tf.mul(SRM,w_init)

          ## define parameters
          self[name_deconv_w] = tf.Variable(tf.transpose(w_init,[1,0,3,2]), name=name_deconv_w,trainable=True)
          self[name_w] = tf.Variable(w_init, name=name_w,trainable=True)

          self[name_b] = tf.Variable(tf.zeros(shape=b_shape),name=name_b,trainable=True)
          self[name_deconv_b] = tf.Variable(tf.zeros(shape=deconv_b_shape),name=name_deconv_b, trainable = True)

          name_w_fixed = name_w + '_fixed'

          ## parameters for after pretraining before fine tuning
          self[name_w_fixed] = tf.Variable(tf.ones(shape),name=name_w_fixed, trainable = False)
          self[name_b+'_fixed'] = tf.Variable(tf.zeros(shape=b_shape),name=name_b+'_fixed', trainable = False)
#
#
      for i in range(self.__num_feed_layers):
#
        name_w = self._weights_str.format(i+1+self.__num_conv_layers)
        name_b = self._biases_str.format(i+1+self.__num_conv_layers)
#
        shape = self.__shape[i+self.__num_conv_layers+1]
        b_shape = [shape[1]]
#
        a = tf.mul(4.0, tf.sqrt(6.0 / (shape[0] + shape[1]) ) )
        self[name_w] = tf.Variable(tf.ones(shape),name=name_w,trainable=True)
        self[name_b] = tf.Variable(tf.zeros(shape=b_shape), name = name_b, trainable=True)

  def _w(self, n, suffix=""):
    return self[self._weights_str.format(n)+suffix]

  def _b(self, n, suffix=""):
    return self[self._biases_str.format(n)+suffix]
#
  def _poolsize(self,n):
    return self.__poolsize[n]
#
  ## only for pre-training i.e when pretreaning sess.run(tf.initialize(scae.get_variables_to_init(n),   while run sess.run(tf.initialize_all_variables()) during fine-tuning
  def get_variables_to_init(self,n):
    ## we take positive index i.e the layer index starts with one
    assert n>0
    assert n<=self.__num_conv_layers+2

    vars_to_init = [self._w(n), self._b(n)]

    if n<= self.__num_conv_layers:
      vars_to_init.append(self._b(n,'_deconv'))
      vars_to_init.append(self._w(n,'_deconv'))

    if 1<n<=self.__num_conv_layers:
      vars_to_init.append(self._w(n-1,'_fixed'))
      vars_to_init.append(self._b(n-1,'_fixed'))
      vars_to_init.append(self._w(n-1))
      vars_to_init.append(self._b(n-1))
#
    return vars_to_init
#
  def pretrain_net(self, input_pl, n, is_target=False):
#
    assert n>0
    assert n <=self.__num_conv_layers
    assign = []
    last_output = input_pl+0
    for i in xrange(n-1):
      w = self._w(i+1,'_fixed')
      b = self._b(i+1,'_fixed')
      assign.append(tf.Variable.assign(w, self._w(i+1)))
      assign.append(tf.Variable.assign(b, self._b(i+1)))
      p = self._poolsize(i+1)
#
      last_output = tf.nn.conv2d(last_output, w, [1,1,1,1],padding='SAME')
      last_output = tf.nn.bias_add(last_output,b)
      last_output = tf.tanh(last_output)/2 + 0.5
#
      last_output = tf.nn.max_pool(last_output,ksize=[1,p,p,1], strides=[1,p,p,1], padding='SAME')
    if is_target:
      return last_output, assign
#
    last_output = tf.nn.conv2d(last_output, self._w(n), [1,1,1,1], padding='SAME')
    last_output = tf.nn.bias_add(last_output, self._b(n))
    last_output = tf.tanh(last_output)/2 + 0.5
#
    p = self._poolsize(n)
    last_output = tf.nn.max_pool(last_output,ksize=[1,p,p,1],strides=[1,p,p,1],padding='SAME')
#
    ## unpool : we used linear interpolation with bicubic

    shape = last_output.get_shape().as_list()
    unpld_height = shape[1]*p
    unpld_width  = shape[2]*p
    out = tf.image.resize_images(last_output,unpld_height,unpld_width,1)
#
    # out = last_output+0
    out = tf.nn.conv2d(out, self._w(n,'_deconv'), [1,1,1,1], padding='SAME')
    out = tf.nn.bias_add(out, self._b(n,'_deconv'))
    out = tf.tanh(out)/2 + 0.5

    return out,assign



#
  def supervised_net(self,input_pl):

    last_output = input_pl+0
    for i in xrange(self.__num_conv_layers):
      w = self._w(i+1,'_fixed')
      b = self._b(i+1,'_fixed')
      p = self._poolsize(i+1)

      last_output = tf.nn.conv2d(last_output, w, [1,1,1,1],padding='SAME')
      last_output = tf.nn.bias_add(last_output,b)
      last_output = tf.tanh(last_output)/2 + 0.5
      last_output = tf.nn.max_pool(last_output,ksize=[1,p,p,1], strides=[1,p,p,1], padding='SAME')

    last_output_shape = last_output.get_shape().as_list()
    out = tf.reshape(last_output, [-1, last_output_shape[1]*last_output_shape[2]*last_output_shape[3]])

    for i in xrange(self.__num_feed_layers):
      w = self._w(i+1+self.__num_conv_layers)
      b = self._b(i+1+self.__num_conv_layers)

      out = tf.tanh(tf.nn.bias_add(tf.matmul(out,w),b))
    y=tf.nn.softmax(out)


    return y

loss_summaries = {}

def training(loss, learning_rate, loss_key=None):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
    loss_key: int giving stage of pretraining so we can store
                loss summaries for each pretraining stage

  Returns:
    train_op: The Op for training.
  """
  if loss_key is not None:
    # Add a scalar summary for the snapshot loss.
    loss_summaries[loss_key] = tf.scalar_summary(loss.op.name, loss)
  else:
    tf.scalar_summary(loss.op.name, loss)
    for var in tf.trainable_variables():
      tf.histogram_summary(var.op.name, var)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op, global_step

def L2_loss(output, target):

  with tf.name_scope("L2_loss"):
    return tf.reduce_sum(tf.square(output - target),name='L2_loss_SCAE')

def loss_supervised(logits,labels):
  batch_size = tf.size(labels)
  NUM_CLASSES = tf.shape(logits)[1]

  labels = tf.expand_dims(labels,1)
  indices = tf.expand_dims(tf.range(0,batch_size),1)
  concated = tf.concat(1, [indices,labels])
  onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size,NUM_CLASSES]),1.0,0.0)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, onehot_labels, name='xentropy')

  loss = tf.reduce_mean(cross_entropy, name = 'xentropy_mean')

  return loss

def main():
 
  with tf.Graph().as_default() as g:
    sess = tf.Session()
    # [(input_ch, input_dim),(first # filter, first filter size, pooling size), (second #filter, second filter size, second pooling size), (.. .. ..),(previous_units=640,hidden_size = 300),(previous_units=300,output_size=2)
    
    fake_data = [ [1,2,3,4], [1,2,3,4], [4,3,2,1], [4,3,2,1]]
    fake_data = np.asarray(fake_data).astype('float32')
    fake_data = np.tile(fake_data,[3,1,1,1])
    fake_data = np.transpose(fake_data,[0,2,3,1])   ## (3,4,4,1)
    fake_label= [0,1,1]
    fake_label = np.asarray(fake_label).astype('int32')
    fake_shape = [(1,4),(1,2,2),(2,2,2),(2,2),(2,2)]
    data_sets = read_data_sets()
    shape = [(1,512),(40,5,4),(10,5,4),(10,5,4),(640,300),(300,2)]
    ae = StackedCAE(shape,sess)
    input_ = tf.placeholder(dtype=tf.float32,shape=(None,shape[0][1],shape[0][1],shape[0][0]))
    labels_= tf.placeholder(dtype=tf.int32, shape = (None))

    out      = {}
    target   = {}
    assign_w = {}
    assign_w_target = {}
    loss     = {}
    training_op = {}
    global_steps= {}
    for i in range(2):
      out[i], assign_w[i] =ae.pretrain_net(input_,i+1,is_target=False)
      target[i], assign_w_target[i] =ae.pretrain_net(input_,i+1,is_target=True)
      loss[i] =  L2_loss(out[i],target[i])
      learning_rate = 0.0001
      training_op[i], global_steps[i] = training(loss[i], learning_rate)
    logits = ae.supervised_net(input_)
    logits_loss = loss_supervised(logits,labels_)
    learning_rate = 100.
    fine_training_op, fine_global_steps = training(logits_loss, learning_rate)
    sess.run(tf.initialize_all_variables())


    for i in range(2):
      execute = [loss[i], global_steps[i], training_op[i]]
      execute.extend(assign_w[i])
      execute.extend(assign_w_target[i])
      for j in range(500):
        pre_data, _ = data_sets.train_ae.next_batch(batch_size)
        output=sess.run(execute,feed_dict={input_: pre_data })
        print(output[0])
        print(output[1])
        print('what happened?')

      print('end')

    for j in range(500):
      fine_data, fine_label = data_sets.train_full.next_batch(batch_size)
      training_summary, losses, steps = sess.run([fine_training_op, logits_loss, fine_global_steps],feed_dict={input_: fine_data, labels_ : fine_label})
      print(losses)
      print(steps)


if __name__=='__main__':
  main()    
