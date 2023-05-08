from abc import ABC, abstractmethod
import numpy as np
from typing import Iterable, Tuple, Union, List
import skimage

class BaseScrambler(ABC):

    def set_state(self, state:dict) -> None:
        self.state = state

    @abstractmethod
    def fit(self, frame:np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def transform(self, frame:np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform(self, frame:np.ndarray) -> np.ndarray:
        pass



class PatchOpMixin:

    def _fit_patch_op(self, frame_size: Union[List,Tuple], patch_size: Union[List,Tuple]) -> None:

        self.image_size = frame_size
        self.patch_size = patch_size

        assert len(self.image_size) == len(self.patch_size) == 2

        # compute pad sizes
        q0, r0 = divmod(self.image_size[0], self.patch_size[0])
        q1, r1 = divmod(self.image_size[1], self.patch_size[1])

        self.pad0 = 0 if r0 == 0 else self.patch_size[0] - r0
        self.pad1 = 0 if r1 == 0 else self.patch_size[1] - r1
        
        self.image_size_after_pad = [
            self.image_size[0] + self.pad0,
            self.image_size[1] + self.pad1
        ]
        
        self.patch_grid_shape = [self.image_size_after_pad[0]//self.patch_size[0], self.image_size_after_pad[1]//self.patch_size[1]]
        self.n_patches = np.prod(self.patch_grid_shape)

    
    # pad the image to align with patch sizes
    def _pad_image(self, img:np.ndarray)->np.ndarray:
        if self.pad0 == self.pad1 == 0:
            return img

        return np.pad(img, [(0, self.pad0), (0, self.pad1), (0,0)])
    
    # crop the valid region from a padded image
    def _crop_image(self, img:np.ndarray)->np.ndarray:
        return img[:self.image_size[0], :self.image_size[1], ...]
    

    def _image_to_patches(self, img:np.ndarray)->np.ndarray:
        # split into rows
        img_row = np.array_split(img, self.patch_grid_shape[0], axis=0)
        img_row_col = [np.array_split(x, self.patch_grid_shape[1], axis=1) for x in img_row]

        return np.concatenate(img_row_col, axis=0)
    
    def _patches_to_image(self, patches:np.ndarray)->np.ndarray:
        # form rows
        patch_rows = np.array_split(patches, self.patch_grid_shape[0], axis=0)
        patch_rows = [np.concatenate(x, axis=1) for x in patch_rows]

        # make images
        image = np.concatenate(patch_rows, axis=0)
        return image
    


class BlockScramble(BaseScrambler, PatchOpMixin):

    def __init__(self, 
                 patch_size:Union[Tuple,List],
                 random_seed:int=42) -> None:
        super().__init__()

        self._patch_size = patch_size
        self.random_seed = random_seed
    
    def _scramble_patches(self, frame:np.ndarray, scrambler:np.ndarray, pad:bool, crop:bool, 
                          return_patches:bool=False)->np.ndarray:
        
        assert frame.dtype == np.uint8

        if pad:
            # first, pad the image
            frame_pad = self._pad_image(frame)
        else:
            frame_pad = frame

        # extract patches
        patches = self._image_to_patches(frame_pad) # (n_patches, patch_height, patch_width, n_channels)

        assert patches.shape[0] == self.n_patches

        # scramble
        new_patches = patches[scrambler]

        if return_patches:
            return new_patches
        
        # rebuild image
        new_image = self._patches_to_image(new_patches)

        if crop:
            return self._crop_image(new_image)
        
        return new_image

    def fit(self, frame:np.ndarray) -> np.ndarray:
        self._fit_patch_op(frame.shape[:2], self._patch_size)

        self.rnd_state = np.random.RandomState(self.random_seed)
        self.patch_order = np.arange(self.n_patches)
        self.rnd_state.shuffle(self.patch_order)

        # build inv transform
        self.inv_patch_order = np.zeros_like(self.patch_order)
        for ind, item in enumerate(self.patch_order):
            self.inv_patch_order[item] = ind


    def transform(self, frame: np.ndarray) -> np.ndarray:
        assert all(map(lambda x: x[0] == x[1], zip(self.image_size, frame.shape[:2])))
        return self._scramble_patches(frame, self.patch_order, pad=True, crop=False)
    
    def inverse_transform(self, frame: np.ndarray) -> np.ndarray:
        assert all(map(lambda x: x[0] == x[1], zip(self.image_size_after_pad, frame.shape[:2])))
        return self._scramble_patches(frame, self.inv_patch_order, pad=False, crop=True)



class MultMod(BaseScrambler):

    def __init__(self)->None:
        super().__init__()
    
    def fit(self, frame:np.ndarray) -> np.ndarray:
        pass

    def transform(self, frame: np.ndarray) -> np.ndarray:
        frame = skimage.img_as_ubyte(frame).astype(np.uint16)
        frame = (frame * 73) % 256
        return frame.astype(np.uint8)
    
    def inverse_transform(self, frame: np.ndarray) -> np.ndarray:
        frame = skimage.img_as_ubyte(frame).astype(np.uint16)
        frame = 255 - ((frame * 7) % 256)
        return frame.astype(np.uint8)



class BlockAdd(BlockScramble):

    def __init__(self, 
                 patch_size:Union[Tuple,List],
                 random_seed:int=42) -> None:
        super().__init__(patch_size=patch_size, random_seed=random_seed)


    def fit(self, frame:np.ndarray) -> np.ndarray:
        super().fit(frame)

        if self.n_patches % 2 != 0:
            raise ValueError('BlockAdd only works when the number of patches is even, consider changing block size')

    def transform(self, frame: np.ndarray) -> np.ndarray:
        assert all(map(lambda x: x[0] == x[1], zip(self.image_size, frame.shape[:2])))

        # first, scramble the patches
        scrambled_patches = self._scramble_patches(frame, self.patch_order, pad=True, crop=False, return_patches=True)

        first_half, second_half = np.array_split(scrambled_patches, 2, axis=0)
        assert first_half.shape[0] == second_half.shape[0]
        
        first_half = first_half.astype(np.float32) / 255.0
        second_half = second_half.astype(np.float32) / 255.0

        # compute the sum
        sum_vals = (first_half + second_half) / 2.0
        diff_vals = (first_half - second_half) / 2.0

        sum_vals = (sum_vals * 255.0).astype(np.uint8)

        # replace the representation of -0.5
        minus_half_mask = np.abs(diff_vals + 0.5) < 1.0e-3
        diff_vals[minus_half_mask] += 1/255.0

        # now, need to switch the encoding for diff vals
        sub_zero_mask = diff_vals < 0
        diff_vals[sub_zero_mask] = 1 + diff_vals[sub_zero_mask]
        diff_vals = (diff_vals * 255.0).astype(np.uint8)

        # checkerboard layout
        new_patches = np.stack([sum_vals, diff_vals], axis=1)
        new_patches = new_patches.reshape((self.n_patches, self.patch_size[0], self.patch_size[1], -1))

        # convert to image
        return self._patches_to_image(new_patches)
    
    def inverse_transform(self, frame: np.ndarray) -> np.ndarray:
        assert all(map(lambda x: x[0] == x[1], zip(self.image_size_after_pad, frame.shape[:2])))
        assert frame.dtype == np.uint8

        # convert to patches
        patches = self._image_to_patches(frame)
        
        # split into two halves
        sum_vals, diff_vals = patches[0::2], patches[1::2]

        # convert to floating point
        sum_vals = sum_vals.astype(np.float32) / 255.0
        diff_vals = diff_vals.astype(np.float32) / 255.0

        # flip the representation of diff_vals
        above_half_mask = diff_vals > 0.5
        diff_vals[above_half_mask] = diff_vals[above_half_mask] - 1

        # compute first and second part
        first_half = sum_vals + diff_vals
        second_half = sum_vals - diff_vals
        
        first_half = np.clip(first_half, 0, 1)
        second_half = np.clip(second_half, 0, 1)

        new_patches = np.concatenate([first_half, second_half], axis=0)
        new_patches = (new_patches * 255.0).astype(np.uint8)
        new_image = self._patches_to_image(new_patches)

        return self._scramble_patches(new_image, self.inv_patch_order, pad=False, crop=True)



class PixelRoll(BaseScrambler):

    def __init__(self, 
                 roll_x:int=1,
                 roll_y:int=1) -> None:
        super().__init__()

        self.roll_x = roll_x
        self.roll_y = roll_y

    
    def fit(self, frame:np.ndarray) -> np.ndarray:
        pass

    def _roll(self, frame: np.ndarray, sign:int) -> np.ndarray:
        ts = self.state['timestamp']
        # compute total x roll
        x_roll = ts * self.roll_x * sign
        y_roll = ts * self.roll_y * sign

        frame = np.roll(frame, shift=x_roll, axis=1)
        frame = np.roll(frame, shift=y_roll, axis=0)

        return frame

    def transform(self, frame: np.ndarray) -> np.ndarray:
        return self._roll(frame, sign=1)
        
    def inverse_transform(self, frame: np.ndarray) -> np.ndarray:
        return self._roll(frame, sign=-1)



if __name__ == '__main__':

    from video_scramble.video_handler import VideoFrameIterator

    vid1 = VideoFrameIterator.from_preset_bigbuckbunny()

    vid_frames = list(vid1)

    scram = BlockScramble([64,64])
    scram.fit(vid_frames[0])

    scrambled_frames = [scram.transform(x) for x in vid_frames]

    vid_new = VideoFrameIterator.from_buf(scrambled_frames)
    vid_new.to_video('scrambled.mp4')

    inv_scramble_frames = [scram.inverse_transform(x) for x in scrambled_frames]

    vid_new = VideoFrameIterator.from_buf(inv_scramble_frames)
    vid_new.to_video('scrambled_inv.mp4')