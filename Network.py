import tensorflow as tf

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

        self._X = tf.placeholder("float", [None, input_units])
        self._encoder_op = self.encoder(self._X)
        self._decoder_op = self.decoder(self._encoder_op)

    def set_loss_optimiser(self):
        y_pred = self._decoder_op
        y_true = self._X
        loss = tf.reduce_mean(tf.pow(y_pred - y_true, 2))
        optimiser = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate).minimize(loss)
        #optimiser = tf.train.MomentumOptimizer(learning_rate=self._learning_rate,momentum =0.99).minimize(loss)
        return loss, optimiser

    def encoder(self, x):
        layer = tf.nn.tanh(tf.add(tf.matmul(x,self._weights['encoder']),self._biases['encoder']))
        return layer
                            
    def decoder(self,x):
        layer = tf.nn.sigmoid(tf.add(tf.matmul(x,self._weights['decoder']),self._biases['decoder']))
        return layer  
        