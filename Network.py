import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import sys
class Network:
    '''
    Class to hold the network
    '''
    def __init__(self, input_units, learing_rate, hidden_units):
        '''
        Initialise a one layer autoencoder
        '''
        self._input_units=input_units 
        self._hidden_units = hidden_units
        self._learning_rate = learing_rate

        # Weights of both encoder and decoder are tranable variables
        self._weights = {
            'encoder':tf.Variable(tf.random_normal([self._input_units, self._hidden_units],mean = 0.0, stddev=0.02)),
            'decoder':tf.Variable(tf.random_normal([self._hidden_units, self._input_units],mean = 0.0, stddev=0.02))
        }

        # Biases of both encoder and decoder are tranable variables
        self._biases = {
            'encoder':tf.Variable(tf.random_normal([self._hidden_units],mean = 0.0, stddev=0.02)),
            'decoder':tf.Variable(tf.random_normal([self._input_units],mean = 0.0, stddev=0.02))
        }

        self._best_weights  = self._weights
        self._best_biases = self._biases


        # Flag to distinguish between training and validation sessions
        self._train_flag = tf.placeholder("bool")
        # Train mask
        self._mask_train = tf.placeholder("float", [None, input_units])
        # Validation mask
        self._mask_val = tf.placeholder("float", [None, input_units])
        # Input
        self._X = tf.placeholder("float", [None, input_units])
        # Apply validation mask only during the validation
        self._encoder_op = tf.cond(self._train_flag, 
                                   lambda: self.encoder(self._X), 
                                   lambda: self.encoder(tf.multiply(self._X,self._mask_val)))

        # Decoder is an output of the encoder
        self._decoder_op = self.decoder(self._encoder_op)

    def set_loss_optimiser(self):
        '''
        Loss function for training (minimised during optimisation)
        '''
        y_pred = self._decoder_op
        y_true = self._X
        # Mean squared error is computed for clean (not corrupted) values only
        loss = tf.divide(tf.reduce_sum(tf.pow(tf.multiply((y_pred-y_true),self._mask_train), 2)),tf.reduce_sum(self._mask_train))
        #optimiser = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate).minimize(loss) # 1e-08
        optimiser = tf.train.AdamOptimizer(learning_rate=self._learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False).minimize(loss)

        #optimiser = tf.train.MomentumOptimizer(learning_rate=self._learning_rate,momentum =0.99).minimize(loss)
        return loss, optimiser

    def set_loss_validation(self):
        '''
        Loss functino for validation
        '''
        y_pred = self._decoder_op
        y_true = self._X
        # Loss is computed only for the validation mask
        loss_val = tf.sqrt(tf.divide(tf.reduce_sum(tf.pow(tf.multiply((y_pred-y_true),(1-self._mask_val)), 2)),tf.reduce_sum(1-self._mask_val)))
        return loss_val


    def save_best(self):
        self._best_weights  = self._weights
        self._best_biases = self._biases


    def set_best_weights(self):
        self._weights  = self._best_weights
        self._biases = self._best_biases

    def encoder(self, x):
        # Operation of the encoder (path through the tanh activation)
        layer = tf.nn.relu(tf.add(tf.matmul(x,self._weights['encoder']),self._biases['encoder']))
        return layer
                            
    def decoder(self,x):
        layer = tf.nn.relu(tf.add(tf.matmul(x,self._weights['decoder']),self._biases['decoder']))
        return layer  
        