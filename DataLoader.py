import tensorflow as tf
import numpy as np
import imageio
import matplotlib.pyplot as plt
import sys
class DataLoader:

    def __init__(self,patch_dims, image_dims):
        self._patch_dims = patch_dims
        self._image_dims = image_dims
        self._image_area = self._image_dims[0]*self._image_dims[1]
        self._patch_area = self._patch_dims[0]*self._patch_dims[1]
        self._image= imageio.imread('./data/0.bmp')
        self._merging_map = np.zeros(self._image_dims, dtype=float)
        self._numb_patches = 0
        self._step = [64,64]#self._patch_dims #self._patch_dims

    def extract_image_patches(self):
        patches = []
        for ii in range(0,self._image.shape[0]-self._step[0],self._step[0]):
            for jj in range(0,self._image.shape[1]-self._step[1],self._step[1]):
                ii_end = (ii+self._patch_dims[0])
                jj_end = (jj+self._patch_dims[1])
                patch = (self._image[ii:ii_end,jj:jj_end]).tolist()
                self._merging_map[ii:ii_end, jj:jj_end] += 1.0
                patches.append(patch)
                self._numb_patches+=1
        patches = np.asarray(patches)

        patches_reshaped = patches.reshape(int(self._numb_patches), self._patch_area)/256.0
        print(patches_reshaped.shape)
        self.combine_image_patches(patches_reshaped)
        return patches_reshaped, self._patch_dims, self._image_dims

    def combine_image_patches(self, patches):
        image = np.zeros(self._image_dims, dtype=int)
        cnt = 0

        patches= patches.reshape(int(self._numb_patches),self._patch_dims[0],self._patch_dims[1])*256
        for ii in range(0,image.shape[0]-self._step[0],self._step[0]):
            for jj in range(0,image.shape[1]-self._step[1],self._step[1]):
                jj_end = (jj+self._patch_dims[1])
                ii_end = (ii+self._patch_dims[0])
                image[ii:ii_end,jj:jj_end] = image[ii:ii_end,jj:jj_end] + np.divide((( np.maximum(
                    np.minimum(np.rint(patches[cnt]),256),0))),(self._merging_map[ii:ii_end, jj:jj_end]))
                cnt = cnt+1

        return image
