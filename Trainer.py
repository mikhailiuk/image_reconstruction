import tensorflow as tf
import sys
import imageio
import copy


class Trainer:
    '''
    Class to perform training, validation and testing. It build with tensorflow
    Thus in the constructor the session is opened, however upon completion the 
    session must be closed with close() method.
    '''

    def __init__(self, net, batch_size, epoches,validation_step):
        '''
        Initialise the trainer
        '''
        
        # Create the network
        self._net = net
        # Set batch
        self._batch_size = batch_size
        # Set number of epoches
        self._epoches = epoches
        # Calculate and display validation error every 100 steps
        self._validation_step = validation_step
        # Minimum validation error - required for finding the best model 
        self._min_error = 99999999999.9

        self._validation_error = []
        # Create tensorflow session and initialise graph and variables
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())
        

    def train(self, dataLoader): 
        '''
        Function to train the network on the dataset from dataLoader
        '''
        # Exctract image patches (split the image and validation masks into patches)
        dataLoader.extract_image_patches()

        # Set the optimiser in the neural network
        loss, optimiser  = self._net.set_loss_optimiser()

        # Iterate over epochs
        for epoch in range(1,self._epoches+1):
            count = 0
            tottal_l = 0.0
            
            # Iterate over the dataset, get batch of image patches, training masks
            # and validation masks
            for batch, btch_msk_tr, btch_msk_vl in dataLoader:
                
                # Run the session with optimiser, get loss. Feeding:
                # 1) Image patches,
                # 2) Training mask
                # 3) Validation mask
                # 4) Set training flag to true (i.e. do not applie validation
                # mask and train on the clean data only)
                _, loss_batch = self._sess.run([optimiser, loss], 
                                                feed_dict={self._net._X:batch,
                                                           self._net._mask_train:btch_msk_tr,
                                                           self._net._mask_val:btch_msk_vl,
                                                           self._net._train_flag:True})

                count=count+1
                tottal_l = tottal_l+loss_batch

            # Shuffle the dataset
            dataLoader.shuffle_order()
            
            # If the first or the display step then validate the network
            if epoch % self._validation_step == 0 or epoch == 1:
                self.validate(dataLoader,epoch)


        self._net.set_best_weights()  

        return self._net

    def validate(self,dataLoader,epoch):
        '''
        Validation function
        '''

        # Set the loss for validation (different from trainig as computed over
        # corrupted examples only with validation masl)
        loss  = self._net.set_loss_validation()
        count = 0
        tottal_l = 0.0

        # Feed all patches and get average loss
        loss_batch = self._sess.run([loss], feed_dict={self._net._X:dataLoader._ptchs,
                                                        self._net._mask_train:dataLoader._ptchs_msk_tr,
                                                        self._net._mask_val:dataLoader._ptchs_msk_vl,
                                                        self._net._train_flag:False})

        # If current loss is less than current minimal 
        if (loss_batch[0]<self._min_error):
            # Set min error to current error
            self._min_error = loss_batch[0]
            # Set new best model
            self._net.save_best()

        self._validation_error.append(loss_batch[0])

        print('Epoch {}: validation loss: {}'.format(epoch,loss_batch[0]))


    def test(self,dataLoader):
        '''
        Test function
        '''


        # To test - feed all patches and get only the output of the network decoder
        g = self._sess.run([self._net._decoder_op], feed_dict={self._net._X:dataLoader._ptchs,
                                                         self._net._mask_train:dataLoader._ptchs_msk_tr,
                                                         self._net._mask_val:dataLoader._ptchs_msk_vl,
                                                         self._net._train_flag:True})
        return g


    def close(self):
        '''
        Need to close the tensorflow session
        '''

        # Close
        self._sess.close()