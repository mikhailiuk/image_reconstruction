import tensorflow as tf
import numpy as np
import imageio
import matplotlib.pyplot as plt
import sys


class DataLoader:

    def __init__(self,patch_dims, image_dims, batch_size, steps):
        self._patch_dims = patch_dims
        self._image_dims = image_dims
        self._batch_size = batch_size
        self._image_area = self._image_dims[0]*self._image_dims[1]
        self._patch_area = self._patch_dims[0]*self._patch_dims[1]
        self._image= imageio.imread('./data/0_noisy.bmp')
        self._mask_train = np.load('./data/0_train_mask.npy')
        self._mask_val = np.load('./data/0_val_mask.npy')
        self._merging_map = np.zeros(self._image_dims, dtype=float)
        self._numb_patches = 0
        self._ordered_arr = None
        self._ptchs = None
        self._ptchs_msk_tr = None
        self._ptchs_msk_vl = None
        self._len_dataset = None
        self._step = steps
        self._index = 0

    def extract_image_patches(self):
        patches = []
        mask_patches_train = []
        mask_patches_val = []
        self._numb_patches = 0
        self._merging_map = np.zeros(self._image_dims, dtype=float)
        cnt = 0
        for ii in range(0,self._image.shape[0]-self._patch_dims[0],self._step[0]):
            ii_end = (ii+self._patch_dims[0])
            for jj in range(0,self._image.shape[1]-self._patch_dims[1],self._step[1]):
                jj_end = (jj+self._patch_dims[1])
                patches.append((self._image[ii:ii_end,jj:jj_end]).tolist())
                mask_patches_train.append((self._mask_train[ii:ii_end,jj:jj_end]).tolist())
                mask_patches_val.append((self._mask_val[ii:ii_end,jj:jj_end]).tolist())
                self._merging_map[ii:ii_end, jj:jj_end] += 1.0
                self._numb_patches+=1
                cnt +=1
                
        patches = np.asarray(patches)
        patches_mask_train = np.asarray(mask_patches_train)
        patches_mask_val = np.asarray(mask_patches_val)

        self._len_dataset = cnt
        self._ordered_arr = np.arange(cnt)
        self._ptchs = patches.reshape(int(self._numb_patches), self._patch_area)/256.0
        self._ptchs_msk_tr = patches_mask_train.reshape(int(self._numb_patches), self._patch_area)
        self._ptchs_msk_vl = patches_mask_val.reshape(int(self._numb_patches), self._patch_area)

        return self._ptchs,self._ptchs_msk_tr, self._ptchs_msk_vl, self._patch_dims, self._image_dims

    def shuffle_order (self):
        self._ordered_arr = np.random.shuffle(self._ordered_arr)

    def combine_image_patches(self, patches):
        image = np.zeros(self._image_dims, dtype=int)
        cnt = 0
        patches = patches.reshape(int(self._numb_patches),self._patch_dims[0],self._patch_dims[1])*256
        for ii in range(0,image.shape[0]-self._patch_dims[0],self._step[0]):
            for jj in range(0,image.shape[1]-self._patch_dims[1],self._step[1]):
                jj_end = (jj+self._patch_dims[1])
                ii_end = (ii+self._patch_dims[0])
                image[ii:ii_end,jj:jj_end] = image[ii:ii_end,jj:jj_end] + np.divide((( np.maximum(
                    np.minimum(np.rint(patches[cnt]),256),0))),(self._merging_map[ii:ii_end, jj:jj_end]))
                image[ii:ii_end,jj:jj_end] = np.multiply(image[ii:ii_end,jj:jj_end],
                                            (1 - self._mask_train[ii:ii_end,jj:jj_end]))+np.multiply(self._image[ii:ii_end,jj:jj_end],self._mask_train[ii:ii_end,jj:jj_end])
                cnt = cnt+1

        return image

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= self._len_dataset:
            self._index = 0
            raise StopIteration

        idx = self._ordered_arr[self._index:(self._index+self._batch_size)]
        self._index = self._index + self._batch_size
        return self._ptchs[idx],self._ptchs_msk_tr[idx],self._ptchs_msk_vl[idx]