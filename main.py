from scipy import misc 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import imageio
from DataLoader import DataLoader
from Network import Network
from Trainer import Trainer


'''
TO RUN:

source ~/tensorflow/bin/activate

THEN
deactivate


TODO:
1) Multiple images
2) Track progress
3) When testing make sure that account for the fact that less informtaion is fed, 
i.e. decrease the magnitude of input inversly proportional to the increase in data information
'''


if __name__ =="__main__":
    # Set parameters
    patch_dims = [32,32]
    image_dims = [512,512]
    steps = [4,4]
    input_units = patch_dims[0]*patch_dims[1]
    learning_rate = 0.001
    batch_size = 128
    epoches = 100
    validation_step = 2
    hiddent_units = 512
    
    # Create dataloader
    dataLoader = DataLoader(patch_dims,image_dims,batch_size,steps)
    
    # Create network
    net = Network(input_units,learning_rate,hiddent_units)

    # Create trainer
    trainer = Trainer(net,batch_size,epoches,validation_step)

    # Train the network
    trainer.train(dataLoader)

    # Test the network and save the image
    g_best = trainer.test(dataLoader)

    dataLoader.combine_and_save_image_patches(g_best[0], 'best')

    # Close the tensorflow session opened in trainer
    trainer.close()
    

