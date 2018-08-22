import tensorflow as tf
import numpy as np
import imageio

class DataLoader:

    def __init__(self,patch_dims, image_dims):
        self._patch_dims = patch_dims
        self._image_dims = image_dims
        self._image_area = self._image_dims[0]*self._image_dims[1]
        self._patch_area = self._patch_dims[0]*self._patch_dims[1]
        self._image= imageio.imread('./lena_gray.bmp')
        self._step = self._patch_dims

    def extract_image_patches(self):
        patches = []
        for ii in range(0,self._image.shape[0],self._step[0]):
            for jj in range(0,self._image.shape[1],self._step[1]):
                patch = (self._image[ii:(ii+self._patch_dims[0]),jj:(jj+self._patch_dims[1])]).tolist()
                patches.append(patch)

        patches = np.asarray(patches)
        print(patches.shape)
        patches_reshaped = patches.reshape(int(self._image_area/self._patch_area), self._patch_area)/256  
        #print(patches_reshaped.shape)    
        return patches_reshaped, self._patch_dims, self._image_dims

    def combine_image_patches(self, patches):
        image = np.empty(self._image_dims, dtype=int)
        cnt = 0
        for ii in range(0,image.shape[0],self._step[0]):
            for jj in range(0,image.shape[1],self._step[1]):
                image[ii:(ii+self._patch_dims[0]),jj:(jj+self._patch_dims[1])] = np.maximum(
                    np.minimum(np.rint(patches[cnt]),256),0)
                cnt = cnt+1
                
        return image
