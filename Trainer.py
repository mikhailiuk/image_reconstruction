import tensorflow as tf

class Trainer:
    def __init__(self):
        self._batch_size = 64
        self._learning_rate = 0.01
        self._num_steps = 10000
        self._display_step = 100

    def train(self, net, dataLoader): 
        patches_reshaped, _ , _ = dataLoader.extract_image_patches()

        loss, optimiser = net.set_loss_optimiser()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for ii in range(1,self._num_steps+1):
                for jj in range(0,(len(patches_reshaped)-self._batch_size),self._batch_size):
                    end = jj+self._batch_size
                    batch = patches_reshaped[jj:end,:]
                    
                    _, l = sess.run([optimiser, loss], feed_dict={net._X:batch})

                if ii % self._display_step == 0 or ii == 1:
                    print('Step %i: Minibatch Loss: %f' % (ii, l))

    def validate(self):
        pass

    def test(self):
        pass
        #g = sess.run([decoder_op], feed_dict={X:patches_reshaped})