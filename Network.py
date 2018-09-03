import tensorflow as tf
import sys
class Network:
    def __init__(self, input_units, learing_rate, hidden_units):
        self._input_units=input_units 
        self._hidden_units = hidden_units
        self._learning_rate = learing_rate
        self._weights = {
            'encoder':tf.Variable(tf.random_normal([self._input_units, self._hidden_units],mean = 0.0, stddev=0.02)),
            'decoder':tf.Variable(tf.random_normal([self._hidden_units, self._input_units],mean = 0.0, stddev=0.02))
        }

        self._biases = {
            'encoder':tf.Variable(tf.random_normal([self._hidden_units],mean = 0.0, stddev=0.02)),
            'decoder':tf.Variable(tf.random_normal([self._input_units],mean = 0.0, stddev=0.02))
        }
        self._train_flag = tf.placeholder("bool")
        self._mask_train = tf.placeholder("float", [None, input_units])
        self._mask_val = tf.placeholder("float", [None, input_units])
        self._X = tf.placeholder("float", [None, input_units])

        self._encoder_op = tf.cond(self._train_flag, 
                                   lambda: self.encoder(self._X), 
                                   lambda: self.encoder(tf.multiply(self._X,self._mask_val)))

        self._decoder_op = self.decoder(self._encoder_op)

    def set_loss_optimiser(self):
        y_pred = self._decoder_op
        y_true = self._X
        loss = tf.divide(tf.reduce_sum(tf.pow(tf.multiply((y_pred-y_true),self._mask_train), 2)),tf.reduce_sum(self._mask_train))
        optimiser = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate).minimize(loss)
        
        #optimiser = tf.train.MomentumOptimizer(learning_rate=self._learning_rate,momentum =0.99).minimize(loss)
        return loss, optimiser

    def set_loss_validation(self):
        y_pred = self._decoder_op
        y_true = self._X
        loss_val = tf.divide(tf.reduce_sum(tf.pow(tf.multiply((y_pred-y_true),(1-self._mask_val)), 2)),tf.reduce_sum(1-self._mask_val))
        return loss_val

    def encoder(self, x):
        layer = tf.nn.tanh(tf.add(tf.matmul(x,self._weights['encoder']),self._biases['encoder']))
        return layer
                            
    def decoder(self,x):
        layer = tf.nn.sigmoid(tf.add(tf.matmul(x,self._weights['decoder']),self._biases['decoder']))
        return layer  
        