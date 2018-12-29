from collections import deque # for computing Train/validation losses are averaged over the last loss_averaging_window
from datetime import datetime
import logging
import os
import pprint as pp

import numpy as np
import tensorflow as tf

from tf_utils import shape_of_tensor

config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True

class TFBaseModel(object):

    """Interface containing some boilerplate code for training tensorflow models.

    Subclassing models must implement self.calculate_loss(), which returns a tensor for the batch loss.
    Code for the training loop, parameter updates, checkpointing, and inference are implemented here and
    subclasses are mainly responsible for building the computational graph beginning with the placeholders
    and ending with the loss tensor.

    Args:
        reader: Class with attributes train_batch_generator, val_batch_generator, and test_batch_generator
            that yield dictionaries mapping tf.placeholder names (as strings) to batch data (numpy arrays).
        batch_size: Minibatch size.
        learning_rate: Learning rate.
        optimizer: 'rms' for RMSProp, 'adam' for Adam, 'sgd' for SGD
        grad_clip: Clip gradients elementwise to have norm at most equal to grad_clip.
        regularization_constant:  Regularization constant applied to all trainable parameters.
        keep_prob: 1 - p, where p is the dropout probability
        early_stopping_steps:  Number of steps to continue training after validation loss has
            stopped decreasing.
        warm_start_init_step:  If nonzero, model will resume training a restored model beginning
            at warm_start_init_step.
        num_restarts:  After validation loss plateaus, the best checkpoint will be restored and the
            learning rate will be halved.  This process will repeat num_restarts times.
        enable_parameter_averaging:  If true, model saves exponential weighted averages of parameters
            to separate checkpoint file.
        min_steps_to_checkpoint:  Model only saves after min_steps_to_checkpoint training steps
            have passed.
        log_interval:  Train and validation accuracies are logged every log_interval training steps.
        loss_averaging_window:  Train/validation losses are averaged over the last loss_averaging_window
            training steps.
        num_validation_batches:  Number of batches to be used in validation evaluation at each step.
        log_dir: Directory where logs are written.
        use_evaluation_metric_as_early_stopping: If true, the early stopping technique will be determined by traget metric u assigned(e.g. f1-score) rather than objective loss.
        checkpoint_dir: Directory where checkpoints are saved.
        prediction_dir: Directory where predictions/outputs are saved.


        So we need an optimization method that can search for hyper-parameters as efficiently as possible, 
        by only evaluating the actual performance when absolutely necessary.

        The priority of hyper-parameters:



    """

    def __init__(
        self,
        reader,
        batch_size=128,
        num_training_steps=20000,
        learning_rate=.01,
        optimizer='adam',
        grad_clip=5,
        regularization_constant=0.0,
        keep_prob=1.0,
        early_stopping_steps=3000,
        warm_start_init_step=0,
        num_restarts=None,
        enable_parameter_averaging=False,
        min_steps_to_checkpoint=100,
        log_interval=20,
        loss_averaging_window=100,
        num_validation_batches=1,
        use_evaluation_metric_as_early_stopping = False,
        log_dir='logs',
        checkpoint_dir='checkpoints',
        prediction_dir='predictions'
    ):
        # args
        self.reader = reader
        self.batch_size = batch_size
        self.num_training_steps = num_training_steps
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.regularization_constant = regularization_constant
        self.warm_start_init_step = warm_start_init_step
        self.early_stopping_steps = early_stopping_steps
        self.keep_prob_scalar = keep_prob
        self.enable_parameter_averaging = enable_parameter_averaging
        self.num_restarts = num_restarts
        self.min_steps_to_checkpoint = min_steps_to_checkpoint
        self.log_interval = log_interval
        self.num_validation_batches = num_validation_batches
        self.loss_averaging_window = loss_averaging_window
        self.use_evaluation_metric_as_early_stopping = use_evaluation_metric_as_early_stopping
        self.log_dir = log_dir
        self.prediction_dir = prediction_dir
        self.checkpoint_dir = checkpoint_dir
        if self.enable_parameter_averaging:
            self.checkpoint_dir_averaged = checkpoint_dir + '_avg'

        # logging
        self.init_logging(self.log_dir)
        logging.info('\nnew run with parameters:\n{}'.format(pp.pformat(self.__dict__)))
        # Build a graph
        self.graph = self.build_graph()
        # Launch the graph in a session
        self.session = tf.Session(graph = self.graph, config = config)
        print ('built graph')

    def calculate_loss(self):
        raise NotImplementedError('subclass must implement this')

    def calculate_evaluation_metric(self):
        raise NotImplementedError('subclass must implement this')

    def fit(self):
        '''
        Performs training with early stopping and halve learning rate 
        '''
        with self.session.as_default():

            if self.warm_start_init_step:
                # reload the model at warm_start_init_step.
                self.restore(self.warm_start_init_step)
                step = self.warm_start_init_step
            else:
                self.session.run(self.init)
                step = 0

            train_generator = self.reader.train_batch_generator(self.batch_size)
            val_generator = self.reader.val_batch_generator(self.num_validation_batches*self.batch_size)
            # Using whole validating set to as measurement of early stopping
            # val_generator = self.reader.val_batch_generator(2403,
            #             shuffle = False,
            #             is_test = True, # allow_smaller_final_batch = True
            #                                     )
            #------------------------
            # initialization of required variable
            #------------------------
            train_loss_history = deque(maxlen=self.loss_averaging_window)
            val_loss_history = deque(maxlen=self.loss_averaging_window)

            best_validation_loss, best_validation_tstep = float('inf'), 0
            restarts = 0

            while step < self.num_training_steps:
                #-----------------------
                # training 
                #-----------------------
                train_batch_df = next(train_generator)
                train_feed_dict = {
                    getattr(self, placeholder_name, None): data
                    for placeholder_name, data in train_batch_df if hasattr(self, placeholder_name)
                }

                train_feed_dict.update({self.learning_rate_var: self.learning_rate})
                if hasattr(self, 'keep_prob'):
                    train_feed_dict.update({self.keep_prob: self.keep_prob_scalar})
                if hasattr(self, 'is_training'):
                    train_feed_dict.update({self.is_training: True})

                train_loss, _ = self.session.run(
                    fetches=[self.loss, self.step],
                    feed_dict=train_feed_dict
                )
                train_loss_history.append(train_loss)
                #-----------------------
                # validating
                #-----------------------
                val_batch_df = next(val_generator)
                # getattr(object, name, default) method returns object whose named attribute's value is to be returned.
                # which is used for input of feed_dict: {tv.Variable_1 or tf.placeholder : external data, }
                val_feed_dict = {
                    getattr(self, placeholder_name, None): data
                    for placeholder_name, data in val_batch_df if hasattr(self, placeholder_name)
                }
                val_feed_dict.update({self.learning_rate_var: self.learning_rate})
                if hasattr(self, 'keep_prob'):
                    val_feed_dict.update({self.keep_prob: 1.0})
                if hasattr(self, 'is_training'):
                    val_feed_dict.update({self.is_training: False})
                #-----------------------
                # determine use which as monitor of early stopping 
                #-----------------------
                if self.use_evaluation_metric_as_early_stopping == True:
                    [val_score] = self.session.run(
                        fetches=[self.score],
                        feed_dict=val_feed_dict
                    )
                    val_loss_history.append(val_score)                   
                else:
                    [val_loss] = self.session.run(
                        fetches=[self.loss],
                        feed_dict=val_feed_dict
                    )
                    val_loss_history.append(val_loss)
                #----------
                # logging: log the training and validating loss every log_interval training steps
                #----------
                if step % self.log_interval == 0:
                    #------------------
                    # calculate batch loss over loss_averaging_window
                    #------------------                    
                    avg_train_loss = sum(train_loss_history) / len(train_loss_history)
                    avg_val_loss = sum(val_loss_history) / len(val_loss_history)
                    if self.use_evaluation_metric_as_early_stopping == True:
                        # the higher score is, the better model is (we assume our score is between 0 and 1)
                        metric_log = (
                            "[[step {:>8}]]     "
                            "[[train]]     loss: {:<12}     "
                            "[[val]]     score: {:<12}     "
                        ).format(step, round(avg_train_loss, 8), round(avg_val_loss, 8))
                        logging.info(metric_log)   
                        # for early stopping
                        avg_val_loss = 1 - avg_val_loss                      
                    else:
                        # the lower loss is, the better model is
                        metric_log = (
                            "[[step {:>8}]]     "
                            "[[train]]     loss: {:<12}     "
                            "[[val]]     loss: {:<12}     "
                        ).format(step, round(avg_train_loss, 8), round(avg_val_loss, 8))
                        logging.info(metric_log)
                    #------------------
                    # early stopping
                    #------------------
                    if avg_val_loss < best_validation_loss:
                        best_validation_loss = avg_val_loss
                        best_validation_tstep = step
                        if step > self.min_steps_to_checkpoint:
                            self.save(step)
                            if self.enable_parameter_averaging:
                                self.save(step, averaged=True)

                    if step - best_validation_tstep > self.early_stopping_steps:

                        if self.num_restarts is None or restarts >= self.num_restarts:
                            if self.use_evaluation_metric_as_early_stopping == True:
                                logging.info('best validation loss of {} at training step {}'.format(
                                    1 - best_validation_loss, best_validation_tstep))
                                
                            else:
                                logging.info('best validation loss of {} at training step {}'.format(
                                    best_validation_loss, best_validation_tstep))
                            logging.info('early stopping - ending training.')
                            return # for terminating this fit function

                        if restarts < self.num_restarts:
                            self.restore(best_validation_tstep)
                            logging.info('halving learning rate')
                            self.learning_rate /= 2.0
                            self.early_stopping_steps /= 2
                            step = best_validation_tstep
                            restarts += 1

                step += 1

            if step <= self.min_steps_to_checkpoint:
                best_validation_tstep = step
                self.save(step)
                if self.enable_parameter_averaging:
                    self.save(step, averaged=True)

            logging.info('num_training_steps reached - ending training')

    def predict(self, chunk_size=216):
        # chunk_size = 2048(by default)
        if not os.path.isdir(self.prediction_dir):
            os.makedirs(self.prediction_dir)
        # check if we have attribute called prediction_tensors
        if hasattr(self, 'prediction_tensors'):
            # prepare the feed_dict to fetch. For exameple, {'tensor_name1':[],...,'tensor_name2':[]}
            prediction_dict = {tensor_name: [] for tensor_name in self.prediction_tensors}

            test_generator = self.reader.test_batch_generator(chunk_size)
            for i, test_batch_df in enumerate(test_generator):
                if i % 100 == 0:
                    print (i*chunk_size)

                test_feed_dict = {
                    getattr(self, placeholder_name, None): data
                    for placeholder_name, data in test_batch_df if hasattr(self, placeholder_name)
                }
                if hasattr(self, 'keep_prob'):
                    # Pracitcally, After finishing the training, it is important to turn off the dropout during development and testing.
                    # Otherwise, the prediction of this model is not stable since dropout add uncertainties to it.
                    test_feed_dict.update({self.keep_prob: 1.0})
                if hasattr(self, 'is_training'):
                    test_feed_dict.update({self.is_training: False})
                # return two tuple of (tensor_name1, tensor_name2, ..., tensor_nameN)
                tensor_names, tf_tensors = zip(*self.prediction_tensors.items())
                # fetech the result from tensor to array
                np_tensors = self.session.run(
                    fetches = tf_tensors,
                    feed_dict = test_feed_dict
                )
                for tensor_name, tensor in zip(tensor_names, np_tensors):
                    # tensor: array
                    prediction_dict[tensor_name].append(tensor)
            #--------------------------------
            # saving the prediction result
            #--------------------------------
            for tensor_name, tensor in prediction_dict.items():
                # tensor: array
                np_tensor = np.concatenate(tensor, 0)
                save_file = os.path.join(self.prediction_dir, '{}.npy'.format(tensor_name))
                logging.info('saving {} with shape {} to {}'.format(tensor_name, np_tensor.shape, save_file))
                np.save(save_file, np_tensor)

        #--------------------------------
        # for obtaining matrix_embedding
        #--------------------------------
        # check if we have attribute called parameter_tensors
        if hasattr(self, 'parameter_tensors'):
            for tensor_name, tensor in self.parameter_tensors.items():
                #  tensor.eval() is shorthand for sess.run(t)
                # the difference between 
                np_tensor = tensor.eval(self.session)

                save_file = os.path.join(self.prediction_dir, '{}.npy'.format(tensor_name))
                logging.info('saving {} with shape {} to {}'.format(tensor_name, np_tensor.shape, save_file))
                np.save(save_file, np_tensor)

    def evaluate(self):
        '''Evaluates performance on whole validating set for hyper-parameters tuning'''
        val_generator = self.reader.val_batch_generator(
            batch_size = self.num_validation_batches*self.batch_size,
            num_epochs = 1, 
            shuffle = False,
            is_test = True, # allow_smaller_final_batch = True
            )
        whole_validating_set_loss = []
        for i, val_batch_df in enumerate(val_generator):
            val_feed_dict = {
                getattr(self, placeholder_name, None): data
                for placeholder_name, data in val_batch_df if hasattr(self, placeholder_name)
            }

            val_feed_dict.update({self.learning_rate_var: self.learning_rate})
            if hasattr(self, 'keep_prob'):
                val_feed_dict.update({self.keep_prob: 1.0})
            if hasattr(self, 'is_training'):
                val_feed_dict.update({self.is_training: False})

            if self.use_evaluation_metric_as_early_stopping == True:
                [val_score] = self.session.run(
                    fetches=[self.score],
                    feed_dict=val_feed_dict
                )
                whole_validating_set_loss.append(val_score)                                
            else:
                [val_loss] = self.session.run(
                    fetches=[self.loss],
                    feed_dict=val_feed_dict
                )
                whole_validating_set_loss.append(val_loss)
        loss_over_whole_validating_set = 1.0 * sum(whole_validating_set_loss) / len(whole_validating_set_loss)
        if self.use_evaluation_metric_as_early_stopping == True:
            logging.info('best score over whole validating set : {}'.format(loss_over_whole_validating_set))
        else:
            logging.info('best loss over whole validating set : {}'.format(loss_over_whole_validating_set))
        return loss_over_whole_validating_set

    def save(self, step, averaged=False):
        '''Save all variables of the TensorFlow graph to file

         Args:
            -session
            -model_path
            -step
        '''
        saver = self.saver_averaged if averaged else self.saver
        checkpoint_dir = self.checkpoint_dir_averaged if averaged else self.checkpoint_dir
        if not os.path.isdir(checkpoint_dir):
            logging.info('creating checkpoint directory {}'.format(checkpoint_dir))
            os.mkdir(checkpoint_dir)

        model_path = os.path.join(checkpoint_dir, 'model')
        logging.info('saving model to {}'.format(model_path))
        #-----------------------
        # Core
        #-----------------------
        saver.save(self.session, model_path, global_step=step)

    def restore(self, step=None, averaged=False):
        '''Reload all the Best variables that were saved to file during optimization'''
        saver = self.saver_averaged if averaged else self.saver
        checkpoint_dir = self.checkpoint_dir_averaged if averaged else self.checkpoint_dir
        if not step:
            # step is None
            model_path = tf.train.latest_checkpoint(checkpoint_dir)
            logging.info('restoring model parameters from {}'.format(model_path))
            saver.restore(self.session, model_path)
        else:
            # step is not None
            model_path = os.path.join(
                checkpoint_dir, 'model{}-{}'.format('_avg' if averaged else '', step)
            )
            logging.info('restoring model from {}'.format(model_path))
            #  It requires a session in which the graph was launched : A Session to use to restore the parameters.
            saver.restore(self.session, model_path)

    def init_logging(self, log_dir):
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
        log_file = 'log_{}.txt'.format(date_str)

        logging.basicConfig(
            filename=os.path.join(log_dir, log_file),
            level=logging.INFO,
            format='[[%(asctime)s]] %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        logging.getLogger().addHandler(logging.StreamHandler())

    def update_parameters(self, loss):
        '''
        It's for optimizing and logging training parameters

        '''
        # create two variable not trainable
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate_var = tf.Variable(0.0, trainable=False)
        # if we need add regularization_constant
        if self.regularization_constant != 0:
            l2_norm = tf.reduce_sum([tf.sqrt(tf.reduce_sum(tf.square(param))) for param in tf.trainable_variables()])
            loss = loss + self.regularization_constant*l2_norm
        # define optimizer
        optimizer = self.get_optimizer(self.learning_rate_var)

        grads = optimizer.compute_gradients(loss)
        # to have norm at most equal to grad_clip
        clipped = [(tf.clip_by_value(g, -self.grad_clip, self.grad_clip), v_) for g, v_ in grads]

        step = optimizer.apply_gradients(clipped, global_step=self.global_step)
        print ('step - whtat optimizer.apply_gradients returns', step)
        if self.enable_parameter_averaging:
            maintain_averages_op = self.ema.apply(tf.trainable_variables())
            with tf.control_dependencies([step]):
                self.step = tf.group(maintain_averages_op)
        else:
            self.step = step

        #--------------
        # logging
        #--------------
        logging.info('all parameters:')
        logging.info(pp.pformat([(var.name, shape_of_tensor(var)) for var in tf.global_variables()]))

        logging.info('trainable parameters:')
        logging.info(pp.pformat([(var.name, shape_of_tensor(var)) for var in tf.trainable_variables()]))

        logging.info('trainable parameter count:')
        logging.info(str(np.sum(np.prod(shape_of_tensor(var)) for var in tf.trainable_variables())))


    def get_optimizer(self, learning_rate, optimizer='adam'):
        '''
        It's for choosing optimizer given learning rate.
        '''
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate)
        elif self.optimizer == 'adagrad':
            return tf.train.AdagradOptimizer(learning_rate)
        elif self.optimizer == 'sgd':
            return tf.train.GradientDescentOptimizer(learning_rate)
        elif self.optimizer == 'rms':
            return tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0.9)
        else:
            #-------
            # standard way to pring the error
            #-------
            assert False, 'optimizer must be adam, adagrad, sgd, or rms'

    def build_graph(self):
        with tf.Graph().as_default() as graph:
            # Create an ExponentialMovingAverage object
            self.ema = tf.train.ExponentialMovingAverage(decay=0.995)
            # Define loss
            self.loss = self.calculate_loss()
            self.update_parameters(self.loss)
            # Define evaluation metric
            if self.use_evaluation_metric_as_early_stopping == True:
                self.score = self.calculate_evaluation_metric()
            # create saver ======> !!!!!!!!Nothing is actually saved at this point!!!!!!!!
            self.saver = tf.train.Saver(max_to_keep=1)
            if self.enable_parameter_averaging:
                self.saver_averaged = tf.train.Saver(self.ema.variables_to_restore(), max_to_keep=1)

            self.init = tf.global_variables_initializer()

            return graph
