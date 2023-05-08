import numpy as np
from typing import Iterator, Iterable, Union, Callable
import skvideo.io, skvideo.datasets
import os

# the video iterator type
class VideoFrameIterator:

    def __init__(self, 
                 data_src:Union[Iterator, Iterable], 
                 src_mode:int='iter',
                 iter_finish_callback:Callable=None,
                 ) -> None:
        
        self.data_src = data_src
        self.src_mode = src_mode
        self.iter_finish_callback = iter_finish_callback

        assert self.src_mode in ('buf', 'iter')

        self.data_iter = self.data_src
        if self.src_mode == 'buf':
            self.data_iter = iter(self.data_iter)
    
    def __iter__(self) -> 'VideoFrameIterator':
        return self
    
    def __next__(self) -> np.ndarray:
        try:
            d = np.asarray(next(self.data_iter))

            if d.ndim == 3:
                assert d.shape[2] == 3
            elif d.ndim == 2:
                d = np.expand_dims(d, axis=-1)
                d = np.real(d, 3, axis=-1)

            return d
        except StopIteration:
            if self.iter_finish_callback is not None:
                self.iter_finish_callback(self)
            raise StopIteration
        
    @staticmethod
    def from_iter(it:iter) -> 'VideoFrameIterator':
        return VideoFrameIterator(it)
    
    @staticmethod
    def from_file(path:str) -> 'VideoFrameIterator':
        vgen = skvideo.io.vreader(path)
        return VideoFrameIterator(vgen, src_mode='iter')
    
    # deletes the file upon iterating all the frames
    @staticmethod
    def from_temp_file(path:str) -> 'VideoFrameIterator':
        # build a callback to delete the temp file
        def delete_tempfile_callback(vit:'VideoFrameIterator', path=path):
            os.remove(path)
        vgen = skvideo.io.vreader(path)
        return VideoFrameIterator(vgen, src_mode='iter', iter_finish_callback=delete_tempfile_callback)
    
    
    @staticmethod
    def from_preset_bigbuckbunny() -> 'VideoFrameIterator':
        return VideoFrameIterator.from_file(skvideo.datasets.bigbuckbunny())
    
    @staticmethod
    def from_preset_bikes() -> 'VideoFrameIterator':
        return VideoFrameIterator.from_file(skvideo.datasets.bikes())
    

    @staticmethod
    def from_buf(buf:Iterable) -> 'VideoFrameIterator':
        return VideoFrameIterator(buf, src_mode='buf')
    
    # save frames to video
    def to_video(self, path:str)->None:
        vwriter = skvideo.io.FFmpegWriter(path)

        for frame in self:
            vwriter.writeFrame(frame)

        vwriter.close()
    

if __name__ == '__main__':

    vit = VideoFrameIterator.from_preset_bigbuckbunny()
    for frame in vit:
        print(frame.shape)