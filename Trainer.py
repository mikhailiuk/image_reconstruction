import tensorflow as tf
import sys
import imageio
class Trainer:
    def __init__(self, net, batch_size, epoches):
        self._net = net
        self._batch_size = batch_size
        self._num_steps = epoches
        self._display_step = 1000
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())

    def train(self, net, dataLoader): 
        patches_reshaped, _,_  = dataLoader.extract_image_patches()
        loss, optimiser = net.set_loss_optimiser()

        for epoch in range(1,self._num_steps+1):
            count = 0
            tottal_l = 0.0
            for jj in range(0,(len(patches_reshaped)-self._batch_size),self._batch_size):
                end = jj+self._batch_size
                batch = patches_reshaped[jj:end,:]
                _, l = self._sess.run([optimiser, loss], feed_dict={self._net._X:batch})
                count=count+1
                tottal_l = tottal_l+l
            if epoch % self._display_step == 0 or epoch == 1:
                
                print('Step {}: Minibatch Loss: {}'.format(epoch, tottal_l/count))

    def validate(self):
        pass

    def test(self,net,dataLoader):
        patches_reshaped, _ , _ = dataLoader.extract_image_patches()
        g = self._sess.run([net._decoder_op], feed_dict={self._net._X:patches_reshaped})
        return g

    def plot_image(self,name, g, dataLoader ):
        image = dataLoader.combine_image_patches(g[0])
        imageio.imsave(name+".png",im=image)

    def close(self):
        self._sess.close()