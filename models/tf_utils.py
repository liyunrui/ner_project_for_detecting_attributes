# ! /usr/bin/env python3
"""
There are some functions that I used to build tensorflow model.

Created on Oct 9 2018

@author: Ray

"""
import os
import tensorflow as tf
from datetime import datetime
import logging

#--------------------
# logging config
#--------------------

def init_logging(log_dir):
    '''
    for recording the experiments.

    log_dir: path
    '''
    #--------------
    # setting
    #--------------
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    log_file = 'log_{}.txt'.format(date_str)
    #--------------
    # config
    #--------------
    logging.basicConfig(
        filename = os.path.join(log_dir, log_file),
        level = logging.INFO,
        format = '[[%(asctime)s]] %(message)s',
        datefmt = '%m/%d/%Y %I:%M:%S %p'
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    

#--------------------
# customized helper function
#--------------------

def shape_of_tensor(tensor, dim=None):
    """
    Get tensor shape/dimension as list/int
    ===========
    Args:
        tensor: tensor in tensorflow
        dim: int, the dimension of this tensor
    """
    if dim is None:
        # return list
        return tensor.shape.as_list()
    else:
        # t
        return tensor.shape.as_list()[dim]

#--------------------
# customized layer
#--------------------

def temporal_convolution_layer(inputs, output_units, convolution_width = 3, dilated = False,
                               causal=False, dilation_rate=[1], bias=True, activation=None, 
                               dropout=None, scope='temporal-convolution-layer', reuse=False):
    """
    Convolution over the temporal axis of sequence data.

    Args:
        inputs: Tensor of shape [batch size, max sequence length, input_units].
        output_units: Output channels for convolution.
        convolution_width: Number of kernel to use in convolution.
        causal: If True, Output at timestep t is a function only depends on inputs where before timestep t. It means no leakage from the future into the past.
        dilated: Simple CNN or Dilated CNN
        dilation_rate:  Dilation rate along temporal axis.
        scope: str.
        reuse: boolean. If we would like to sharing this weight.

    Returns:
        Tensor of shape [batch size, max sequence length, output_units].

    """
    with tf.variable_scope(scope, reuse=reuse):
        if dilated == True:
            #-------------------
            # dilated CNN
            #-------------------
            if causal:
                # comput how many zeros we need padded given dilation rate(d) and convolution_width(k)
                shift = (convolution_width - 1) * dilation_rate[0]
                pad = tf.zeros([tf.shape(inputs)[0], shift, inputs.shape.as_list()[2]])
                # pad zeros over temporal axis at the left side of inputs
                inputs = tf.concat([pad, inputs], axis=1)
                # filter weight
                W = tf.get_variable(
                    name='weights',
                    initializer=tf.contrib.layers.variance_scaling_initializer(),
                    shape=[convolution_width, shape_of_tensor(inputs, 2), output_units]
                )
                # convolution = matrix multiplication + element-wise addition
                z = tf.nn.convolution(inputs, W, padding='VALID', dilation_rate=dilation_rate)
                # adding bias if True
                if bias:
                    b = tf.get_variable(
                        name='biases',
                        initializer=tf.constant_initializer(),
                        shape=[output_units]
                    )
                    z = z + b
                # adding non-linear output if True
                z = activation(z) if activation else z
                # adding dropout if True
                z = tf.nn.dropout(z, dropout) if dropout is not None else z
            else:
                # filter weight
                W = tf.get_variable(
                    name='weights',
                    initializer=tf.contrib.layers.variance_scaling_initializer(),
                    shape=[convolution_width, shape_of_tensor(inputs, 2), output_units]
                )
                # convolution = matrix multiplication + element-wise addition
                z = tf.nn.convolution(inputs, W, padding='SAME', dilation_rate=dilation_rate)
                # adding bias if True
                if bias:
                    b = tf.get_variable(
                        name='biases',
                        initializer=tf.constant_initializer(),
                        shape=[output_units]
                    )
                    z = z + b
                # adding non-linear output if True
                z = activation(z) if activation else z
                # adding dropout if True
                z = tf.nn.dropout(z, dropout) if dropout is not None else z                
        else:
            #-------------------
            # simple CNN
            #-------------------
            # filter weight
            W = tf.get_variable(
                name='weights',
                initializer=tf.contrib.layers.variance_scaling_initializer(),
                # shape = spatial_filter_shape + [in_channels, out_channels]
                shape=[convolution_width, shape_of_tensor(inputs, 2), output_units]
            )
            # convolution = matrix multiplication + element-wise addition
            z = tf.nn.convolution(input = inputs, 
                          filter = W, 
                          padding='SAME', 
                          )
            if bias:
                b = tf.get_variable(
                    name='biases',
                    initializer=tf.constant_initializer(),
                    shape=[output_units]
                )
                z = z + b

            # adding non-linear output if True
            z = activation(z) if activation else z
            # adding dropout if True
            z = tf.nn.dropout(z, dropout) if dropout is not None else z
        return z

def time_distributed_dense_layer(inputs, output_units, bias=True, activation=None, batch_norm=None,
                                 dropout=None, scope='time-distributed-dense-layer', reuse=False):
    """
    Applies a shared dense layer to each timestep of a tensor of shape [batch_size, max_seq_len, input_units]
    to produce a tensor of shape [batch_size, max_seq_len, output_units].

    ===========
    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        output_units: Number of output units.
        activation: activation function.
        dropout: A scalar Tensor, keep prob.

    Returns:
        Tensor of shape [batch size, max sequence length, output_units].

    """
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[shape_of_tensor(inputs, -1), output_units]  #[input_dim on feature axis, output_units]
        )
        # matrix multiplication
        z = tf.einsum('ijk,kl->ijl', inputs, W)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b
        # doing batch_norm before activation is better for training nn.
        if batch_norm is not None:
            z = tf.layers.batch_normalization(z, training=batch_norm, reuse=reuse)
        # activation fisst and before otput doing dropout
        z = activation(z) if activation else z
        z = tf.nn.dropout(z, keep_prob = dropout) if dropout is not None else z
        return z

#----------------
# Generic Convolutional Network for sequence modeling
#-----------------

def weightNorm_temporal_convolution_layer(inputs, output_units, convolution_width = 3, causal = True, 
                                          dilation_rate=[1], bias=True, activation = tf.nn.relu, dropout=None, 
                                          scope='weightNorm-temporal-convolution-layer', reuse=False, 
                                          gated = False):
    """
    Reference:
        - Implement Weight Normarlization [Salimans 16]
        - Implement Spatial Dropout[LeCun 15]
            It's a special dropout used in convolutional layer.
        - Implement Gating Mechanisms [Dauphin 17], (Not in TCN paper)
    ------------
    Args:
        inputs: Tensor of shape [batch size, max sequence length, input_units].
        output_units: Output unit for each convolution layer. 
        convolution_width: Number of kernel to use in convolution layer.
        causal: If True, Output at timestep t is a function only depends on inputs where before timestep t. It means no leakage from the future into the past.
        dropout: A scalar Tensor, keep prob.
        gated: (not in original paper) use gated linear unit as activation
    Returns
        A tensor of shape [batch size, max sequence length, num_channels[-1]]
   

    """
    with tf.variable_scope(scope, reuse=reuse):
        if gated:
            output_units = output_units * 2
        
        if causal:
            # comput how many zeros we need padded given dilation rate(d) and convolution_width(k)
            shift = (convolution_width - 1) * dilation_rate[0]
            pad = tf.zeros([tf.shape(inputs)[0], shift, inputs.shape.as_list()[2]])
            # pad zeros over temporal axis at the left side of inputs
            inputs = tf.concat([pad, inputs], axis=1)
            # filter weight
            V = tf.get_variable(
                name='weights',
                initializer=tf.contrib.layers.variance_scaling_initializer(),
                shape=[convolution_width, shape_of_tensor(inputs, 2), output_units]
            )
            g = tf.get_variable('g', shape=[output_units], dtype=tf.float32,
                                initializer=tf.constant_initializer(1.), trainable=True)
            # use weight normalization (Salimans & Kingma, 2016): normalizing the parameter vector along dim affecting scalar output of each neuron.
            W = tf.reshape(g, [1, 1, output_units]) * tf.nn.l2_normalize(V, dim = [0, 1]) # normalizaing along convolution_width and input channels controlling each 
            # convolution = matrix multiplication + element-wise addition
            z = tf.nn.convolution(inputs, W, padding='VALID', dilation_rate=dilation_rate)
            # adding bias if True
            if bias:
                b = tf.get_variable(
                    name='biases',
                    initializer=tf.constant_initializer(),
                    shape=[output_units]
                )
                z = z + b
            # adding non-linear output if True
            if gated:
                # use gated linear units (Dauphin 2016) as activation
                split0, split1 = tf.split(z, num_or_size_splits=2, axis=2)
                split1 = tf.sigmoid(split1)
                z = tf.multiply(split0, split1)
            else:
                z = activation(z) if activation else z
            # adding spatial dropout if True
            if dropout is not None:
                # at each training step, a whole channel is zeroed out.(feature-axis)
                # Here, what spatial dropout did is like feature fraction in tree-based model: random select part of features duriing training step.
                noise_shape = (tf.shape(z)[0], tf.constant(1), tf.shape(z)[2]) #[batch_size,1, output_units]
                z = tf.nn.dropout(z, dropout, noise_shape)
            else:
                pass
        else:
            # filter weight
            V = tf.get_variable(
                name='weights',
                initializer=tf.contrib.layers.variance_scaling_initializer(),
                shape=[convolution_width, shape_of_tensor(inputs, 2), output_units]
            )
            g = tf.get_variable('g', shape=[output_units], dtype=tf.float32,
                                initializer=tf.constant_initializer(1.), trainable=True)
            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1, 1, output_units]) * tf.nn.l2_normalize(V, dim = [0, 1])
            # convolution = matrix multiplication + element-wise addition
            z = tf.nn.convolution(inputs, W, padding='SAME', dilation_rate=dilation_rate)
            # adding bias if True
            if bias:
                b = tf.get_variable(
                    name='biases',
                    initializer=tf.constant_initializer(),
                    shape=[output_units]
                )
                z = z + b
            # adding non-linear output if True
            if gated:
                # use gated linear units (Dauphin 2016) as activation
                split0, split1 = tf.split(z, num_or_size_splits=2, axis=2)
                split1 = tf.sigmoid(split1)
                z = tf.multiply(split0, split1)
            else:
                z = activation(z) if activation else z
            # adding spatial dropout if True
            if dropout is not None:
                # at each training step, a whole channel is zeroed out.(feature- axis)
                # Here, what spatial dropout did is like feature fraction in tree-based model: random select part of features duriing training step.
                noise_shape = (tf.shape(z)[0], tf.constant(1), tf.shape(z)[2]) #[batch_size,1, output_units]
                z = tf.nn.dropout(z, dropout, noise_shape)
            else:
                pass
    return z

def ResidualBlock(input_layer, out_channels, convolution_width, dilation_rate, num_block, causal = True,
                  dropout = None, atten=False, use_highway=False, gated=False, scope='residual-block', 
                  reuse=False):
    """
    Residual Block in TCN (Bai 2018)
    
    Here, we also implement another short connections, alternative to Residual Network, 
    called Highway Networks [Srivastava 15]
    ----------------
    Arguments
        input_layer: A tensor of shape [N, L, Cin]
        out_channels: output dimension
        convolution_width: Number of kernel to use in convolution layer.
        dilation_rate: holes inbetween
        dropout: prob. to drop weights
        num_block: int. the n-th residual block.
        atten: (not in original paper) add self attention block after Conv.
        use_highway: (not in original paper) use highway as residual connection
        gated: (not in original paper) use gated linear unit as activation
    Returns
        A Tensor of shape [batch size, max sequence length, out_channels].
    """
    with tf.variable_scope(scope, reuse=reuse):
        # for short connetions
        in_channels = input_layer.get_shape()[-1]
        #-------------------
        # Layer1: Dilated Causal Conv + WeightNorm
        #-------------------
        conv1 = weightNorm_temporal_convolution_layer(
                inputs = input_layer, 
                output_units = out_channels,
                convolution_width = convolution_width,
                dilation_rate = [dilation_rate],
                causal = causal,
                bias=True,
                activation=None, 
                dropout=dropout,
                scope='weightNorm-cnn-layer-1-in-block-{}'.format(num_block+1),
                reuse=False,
                gated=gated,
                        )
        if atten:
            conv1 = attentionBlock(conv1, dropout)
        #-------------------
        # Layer2: Dilated Causal Conv + WeightNorm
        #-------------------
        conv2 = weightNorm_temporal_convolution_layer(
                inputs = conv1, 
                output_units = out_channels,
                convolution_width = convolution_width,
                dilation_rate = [dilation_rate],
                causal = causal,
                bias=True,
                activation=None, 
                dropout=dropout,
                scope='weightNorm-cnn-layer-2-in-block-{}'.format(num_block+1),
                reuse=False,
                gated=gated,
                        )
        if atten:
            conv2 = attentionBlock(conv2, dropout)


        # highway connetions or residual connection
        residual = None
        if use_highway:
            #---------------
            # use highway network as short connections
            #---------------
            W_h = tf.get_variable('W_h-in-block-{}'.format(num_block+1), [1, int(input_layer.get_shape()[-1]), out_channels],
                                  tf.float32, tf.random_normal_initializer(0, 0.01), trainable=True)
            b_h = tf.get_variable('b_h-in-block-{}'.format(num_block+1), shape=[out_channels], dtype=tf.float32,
                                  initializer=None, trainable=True)
            H = tf.nn.bias_add(tf.nn.convolution(input_layer, W_h, 'SAME'), b_h)

            W_t = tf.get_variable('W_t-in-block-{}'.format(num_block+1), [1, int(input_layer.get_shape()[-1]), out_channels],
                                  tf.float32, tf.random_normal_initializer(0, 0.01), trainable=True)
            b_t = tf.get_variable('b_t-in-block-{}'.format(num_block+1), shape=[out_channels], dtype=tf.float32,
                                  initializer=None, trainable=True)
            T = tf.nn.bias_add(tf.nn.convolution(input_layer, W_t, 'SAME'), b_t)
            T = tf.nn.sigmoid(T)
            residual = H*T + input_layer * (1.0 - T)
        elif in_channels != out_channels:
            #---------------
            # use residual network as short connections
            #---------------
            # we only use 1 filter aka convolution_width = 1 from the perspective of 1-D convolutions.
            W_h = tf.get_variable('W_h-in-block-{}'.format(num_block+1), [1, int(input_layer.get_shape()[-1]), out_channels],
                                  tf.float32, tf.random_normal_initializer(0, 0.01), trainable=True)
            b_h = tf.get_variable('b_h-in-block-{}'.format(num_block+1), shape=[out_channels], dtype=tf.float32,
                                  initializer=None, trainable=True)
            residual = tf.nn.bias_add(tf.nn.convolution(input_layer, W_h, 'SAME'), b_h)
        else:
            print("There is no need to use additional 1x1 convolution to solve discrepant input-output widths")

        res = input_layer if residual is None else residual

        return tf.nn.relu(conv2 + res)

def TemporalConvNet(inputs, num_channels, convolution_width=3, causal = True,
                    dropout= None, atten=False, use_highway=False, use_gated=False, scope = 'TCN'):
    """
    A stacked dilated CNN architecture described in Bai 2018
    ------------
    Args:
        inputs: Tensor of shape [batch size, max sequence length, input_units].
        num_channels: List of output channels aka output unit for each convolution layer. 
        convolution_width: Number of kernel to use in convolution layer.
        causal: If True, Output at timestep t is a function only depends on inputs where before timestep t. It means no leakage from the future into the past.
        dropout: A scalar Tensor, keep prob.
        atten: (not in original paper) add self attention block after Conv.
        use_highway: (not in original paper) use highway as residual connection
        gated: (not in original paper) use gated linear unit as activation
    Returns
        A tensor of shape [batch size, max sequence length, num_channels[-1]]
    Notice:
        len(num_channels) = how many residual block we used in TCN.
    ------------
    Reference:
        - https://github.com/YuanTingHsieh
    """
    num_residual_block = len(num_channels)
    for i in range(num_residual_block):
        dilation_factor = 2 ** i # each residual block use same convolution_width and dilation factor 
        out_channels = num_channels[i]
        inputs = ResidualBlock(
                            input_layer = inputs, 
                            out_channels = out_channels, 
                            convolution_width = convolution_width, 
                            dilation_rate=dilation_factor,
                            num_block = i,
                            causal = causal,
                            dropout=dropout, 
                            atten=atten, 
                            use_highway = use_highway,
                            gated=use_gated,
                            scope = scope
                                   )

    return inputs

#------------------
# customized objective loss
#------------------
def sequence_softmax_loss(y, y_hat, sequence_lengths, max_sequence_length):
    """
    Calculates average softmax-cross-entropy on variable length sequences.

    Args:
        y: Label tensor of shape [batch size, max_sequence_length], which should be indices of label.
        y_hat: Prediction tensor, [batch size, max_sequence_length, num_class], which should be unscaled score.
        sequence_lengths: Sequence lengths.  Tensor of shape [batch_size].
        max_sequence_length: maximum length of padded sequence tensor.

    Returns:
        batch_softmax_loss. 0-dimensional tensor.
        
    Reference of sparse_softmax_cross_entropy_with_logits:
        -Official: https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits
        -Blog : https://blog.csdn.net/yc461515457/article/details/77861695
    """
    # softmax cross-entropy between y(y_true) and y_hat(y_pred)
    softmax_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits= y_hat) # (?, max_sequence_length)
    # returns a boolean mask tensor for the first N positions of each cell.
    sequence_mask = tf.sequence_mask(lengths = sequence_lengths, maxlen=max_sequence_length) # (?, max_sequence_length)
    # convert boolean into 1 or 0
    sequence_mask = tf.cast(sequence_mask, tf.float32) # (?, max_sequence_length)
    # compute sum of loss over each timestep / number of real training example of this batch (aka batch loss)
    batch_softmax_loss = tf.reduce_sum(softmax_losses*sequence_mask) / tf.cast(tf.reduce_sum(sequence_lengths), tf.float32)
    return batch_softmax_loss

#------------------
# customized evaluation metric
#------------------

def sequence_evaluation_metric(y, y_hat, sequence_lengths, max_sequence_length):
    """
    Calculates average evaluation metric on variable length sequences. 

    Args:
        y: Label tensor of shape [batch size, max_sequence_length], which should be index of label.(y_true)
        y_hat: Prediction tensor, [batch size, max_sequence_length], which should be index of predicted label.(y_pred)
        sequence_lengths: Sequence lengths.  Tensor of shape [batch_size].
    Returns:
        metrics: dict. metrics["f1"] = 0.72 on each batch
    """
    #---------------
    # calculate
    #---------------
    
    # step1: element-wise comparison between y_true and y_pred
    correct_preds = tf.cast(tf.equal(y, y_hat), tf.float32) # Returns boolean tensor where the truth value of (x == y) element-wise.
    # returns a boolean mask tensor for the first N positions of each cell.
    sequence_mask = tf.sequence_mask(lengths = sequence_lengths, maxlen=max_sequence_length) # (?, max_sequence_length)
    # convert boolean into 1 or 0
    sequence_mask = tf.cast(sequence_mask, tf.float32) # (?, max_sequence_length)

    # step2: dynamically consider the case their timestep do not pass his history lenghth
    correct_preds = correct_preds*sequence_mask
    y = tf.cast(y, dtype = tf.float32)*sequence_mask
    y_hat = tf.cast(y_hat,dtype = tf.float32)*sequence_mask
    # step3: tf.reduce_sum on temporal axi
    correct_preds = tf.reduce_sum(correct_preds, axis = 1)
    # step4: element-wise comparison between correct_preds and sequence_lengths
    correct_preds = tf.cast(tf.equal(tf.cast(correct_preds, tf.int32), sequence_lengths), tf.float32) # (batch_size,), which element representing if this sentence is correct or not
    # Final step: calculate scalar
    correct_preds = tf.cast(tf.count_nonzero(correct_preds, axis = None), tf.float32)
    total_correct = tf.cast(tf.count_nonzero(tf.count_nonzero(y, axis = 1), axis = None), tf.float32)
    total_preds = tf.cast(tf.count_nonzero(tf.count_nonzero(y_hat, axis = 1), axis = None), tf.float32) # 0-D, number of our rediction which is non-zero

    #---------------
    # output
    #---------------
    p = tf.cond(tf.greater(correct_preds, tf.zeros(shape=[])), lambda : correct_preds/total_preds, lambda: tf.zeros(shape=[]))
    r = tf.cond(tf.greater(correct_preds, tf.zeros(shape=[])), lambda : correct_preds/total_correct, lambda: tf.zeros(shape=[]))
    f1 = tf.cond(tf.greater(correct_preds, tf.zeros(shape=[])), lambda : 2 * p * r / (p + r), lambda: tf.zeros(shape=[]))
    acc = correct_preds/ tf.cast(tf.shape(y)[0], dtype = tf.float32) # correct_preds/ batch_size
    return {"acc": 100*acc, "precision": p, "recall": r, "f1": f1}


