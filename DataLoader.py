import tensorflow as tf
import numpy as np
import imageio
import matplotlib.pyplot as plt
import sys


class DataLoader:
    '''
    DataLoader class to read, hold and process the data. Contains method to 
    itterate over. 
    '''

    def __init__(self,patch_dims, image_dims, batch_size, steps):
        '''
        Initialise
        '''
        self._patch_dims = patch_dims
        self._image_dims = image_dims
        self._batch_size = batch_size
        self._image_area = self._image_dims[0]*self._image_dims[1]
        self._patch_area = self._patch_dims[0]*self._patch_dims[1]
        # Set the image and patches
        self._image= imageio.imread('./data/0_noisy.bmp')
        self._mask_train = np.load('./data/0_train_mask.npy')
        self._mask_val = np.load('./data/0_val_mask.npy')
        # Need map with weights to combine image together if use overlapping patches 
        self._merging_map = np.zeros(self._image_dims, dtype=float)
        # Order in whitch patches are fed to the network
        self._ordered_arr = None
        # Numpy array with image patches
        self._ptchs = None
        # Numpy array with training mask (lockation of missing values)
        self._ptchs_msk_tr = None
        # Numpy array with validation mask values
        self._ptchs_msk_vl = None
        # Length of the dataset (number of patches)
        self._len_dataset = 0
        # Step - shift window for the patch (contains tupple x and y) 
        self._step = steps
        # If itterating over the dataset - the starting index in the 
        # self._ordered_arr
        self._index = 0

    def extract_image_patches(self):
        '''
        Function to extract image patches
        '''

        # Create initial lists to hold patches
        patches = []
        mask_patches_train = []
        mask_patches_val = []
        # Create numpy array for the merging map
        self._merging_map = np.zeros(self._image_dims, dtype=float)

        # Iterate over the first dimension of the image with an x-step
        for ii in range(0,self._image.shape[0]-self._patch_dims[0],self._step[0]):
            # Ending index for the current patch extracted from the first fimensions
            ii_end = (ii+self._patch_dims[0])
            # Iterate over the second dimension of the image with an x-step
            for jj in range(0,self._image.shape[1]-self._patch_dims[1],self._step[1]):
                
                # Ending index for the current patch extracted from the second fimensions
                jj_end = (jj+self._patch_dims[1])

                # Append extracted patches from the image and noise masks to their
                # lists
                patches.append((self._image[ii:ii_end,jj:jj_end]).tolist())
                mask_patches_train.append((self._mask_train[ii:ii_end,jj:jj_end]).tolist())
                mask_patches_val.append((self._mask_val[ii:ii_end,jj:jj_end]).tolist())
                self._merging_map[ii:ii_end, jj:jj_end] += 1.0
                self._len_dataset +=1

        # Convert lists to arrays        
        patches = np.asarray(patches)
        patches_mask_train = np.asarray(mask_patches_train)
        patches_mask_val = np.asarray(mask_patches_val)

        # Fill the orderrred array
        self._ordered_arr = np.arange(self._len_dataset)

        # Reshape np arrays with data and scale the image between 0 and 1
        self._ptchs = patches.reshape(int(self._len_dataset), self._patch_area)/256.0
        self._ptchs_msk_tr = patches_mask_train.reshape(int(self._len_dataset), self._patch_area)
        self._ptchs_msk_vl = patches_mask_val.reshape(int(self._len_dataset), self._patch_area)

        return self._ptchs,self._ptchs_msk_tr, self._ptchs_msk_vl, self._patch_dims, self._image_dims

    def shuffle_order (self):
        '''
        Function to shuffle the data
        '''

        self._ordered_arr = np.random.shuffle(self._ordered_arr)

    def combine_image_patches(self, patches):
        '''
        Function to combine reconstructed patches into an image
        '''

        # Image to fill
        image = np.zeros(self._image_dims, dtype=int)
        cnt = 0
        # Reshpe and rescale patches
        patches = patches.reshape(int(self._len_dataset),self._patch_dims[0],self._patch_dims[1])*256

        # Go over the first image dimension with a step
        for ii in range(0,image.shape[0]-self._patch_dims[0],self._step[0]):
            
            # End index in the first dimension
            ii_end = (ii+self._patch_dims[0])

            # Go over the second dimension 
            for jj in range(0,image.shape[1]-self._patch_dims[1],self._step[1]):

                # End index in the second dimension
                jj_end = (jj+self._patch_dims[1])
                image[ii:ii_end,jj:jj_end] = image[ii:ii_end,jj:jj_end] + np.divide((( np.maximum(
                    np.minimum(np.rint(patches[cnt]),256),0))),(self._merging_map[ii:ii_end, jj:jj_end]))
                image[ii:ii_end,jj:jj_end] = np.multiply(image[ii:ii_end,jj:jj_end],
                                            (1 - self._mask_train[ii:ii_end,jj:jj_end]))+np.multiply(self._image[ii:ii_end,jj:jj_end],self._mask_train[ii:ii_end,jj:jj_end])
                cnt = cnt+1

        return image

    def __iter__(self):
        '''
        To iterate over the class need method __iter__ which would return an 
        object with __next__ method
        '''
        return self

    def __next__(self):
        '''
        Function to return the next value if called from the for loop
        '''

        # If current index is not less than length of the dataset, stop
        if self._index >= self._len_dataset:
            self._index = 0
            raise StopIteration

        # Find index from the ordered_arr - holds order of the patches and is 
        # shuffled from time to time
        idx = self._ordered_arr[self._index:(self._index+self._batch_size)]

        # Update current index by the size of the batch
        self._index = self._index + self._batch_size
        return self._ptchs[idx],self._ptchs_msk_tr[idx],self._ptchs_msk_vl[idx]