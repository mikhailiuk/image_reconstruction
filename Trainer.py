import tensorflow as tf
import sys
import imageio
class Trainer:
    def __init__(self, net, batch_size, epoches):
        self._net = net
        self._batch_size = batch_size
        self._num_steps = epoches
        self._display_step = 100
        self._sess = tf.Session()
        self._best_model = net
        self._min_error = 99999999999.9
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())

    def train(self, dataLoader): 

        dataLoader.extract_image_patches()
        loss, optimiser  = self._net.set_loss_optimiser()

        for epoch in range(1,self._num_steps+1):
            count = 0
            tottal_l = 0.0
            for batch, btch_msk_tr, btch_msk_vl in dataLoader:
                _, loss_batch = self._sess.run([optimiser, loss], feed_dict={self._net._X:batch,
                                                                    self._net._mask_train:btch_msk_tr,
                                                                    self._net._mask_val:btch_msk_vl,
                                                                    self._net._train_flag:True})

                count=count+1
                tottal_l = tottal_l+loss_batch

            #dataLoader.shuffle_order()
            if epoch % self._display_step == 0 or epoch == 1:
                self.validate(dataLoader,epoch)
                #print('Step {}: Minibatch Loss: {}'.format(epoch, tottal_l/count))

    def validate(self,dataLoader,epoch):
        loss  = self._net.set_loss_validation()
        count = 0
        tottal_l = 0.0
        loss_batch = self._sess.run([loss], feed_dict={self._net._X:dataLoader._ptchs,
                                                        self._net._mask_train:dataLoader._ptchs_msk_tr,
                                                        self._net._mask_val:dataLoader._ptchs_msk_vl,
                                                        self._net._train_flag:False})

        if (loss_batch[0]<self._min_error):
            self._min_error = loss_batch[0]
            self._best_model = self._net

        print('Step {}: Minibatch Loss: {}'.format(epoch,loss_batch[0]))


    def test(self,dataLoader):
        
        g = self._sess.run([self._best_model._decoder_op], feed_dict={self._best_model._X:dataLoader._ptchs,
                                                         self._best_model._mask_train:dataLoader._ptchs_msk_tr,
                                                         self._best_model._mask_val:dataLoader._ptchs_msk_vl,
                                                         self._best_model._train_flag:True})
        return g

    def plot_image(self,name, g, dataLoader ):
        image = dataLoader.combine_image_patches(g[0])
        imageio.imsave(name+".png",im=image)

    def close(self):
        self._sess.close()