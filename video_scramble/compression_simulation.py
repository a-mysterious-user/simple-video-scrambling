# simulates the effect of video compression

from video_scramble.video_handler import VideoFrameIterator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import subprocess
import tempfile
import skimage
import numpy as np

class CompressionSimulator(ABC):
    
    # simulates compression over the input video iterator
    # returns the frames after compression
    @abstractmethod
    def transform(self, vid:VideoFrameIterator) -> VideoFrameIterator:
        pass


# for now, only supports H.264 CRF mode
@dataclass
class H264CompressionParam:
    preset:str = 'fast'
    crf:int = 30


# performs the H.264 compression over the input sequence
class H264Compression(CompressionSimulator):
    
    def __init__(self, 
                 params:List[H264CompressionParam], # the H.264 compression parameters of each pass,
                 pipe_chunksize:int=1024,
                 verbose:bool=True
                 ) -> None:
        super().__init__()

        self.params = params
        assert len(self.params) > 0

        self.pipe_chunksize = 1024
        self.verbose = verbose

    def transform(self, vid: VideoFrameIterator) -> VideoFrameIterator:
        current_vid = vid

        for param in self.params:
            # get a temp filename
            tempfile_obj = tempfile.NamedTemporaryFile(suffix='.mp4')
            tempfile_path = tempfile_obj.name
            tempfile_obj.close()

            # we need the first frame to acquire the frame size
            first_frame = next(current_vid)
            
            # run ffmpeg to compress the video
            cp = subprocess.Popen(
                ['ffmpeg', '-hide_banner', '-y', '-an', 
                '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                 '-s', '{}x{}'.format(first_frame.shape[1], first_frame.shape[0]), '-i', '-',
                 '-c:v', 'libx264', '-preset', param.preset, '-crf', str(param.crf),
                 tempfile_path
                ],
                stdin=subprocess.PIPE,
                stdout=None if self.verbose else subprocess.DEVNULL,
                stderr=None if self.verbose else subprocess.DEVNULL,
            )

            # pass the frames to ffmpeg using PIPE
            while True:

                if first_frame is not None:
                    frame = first_frame
                    first_frame = None
                else:
                    try:
                        frame = next(current_vid)
                    except StopIteration:
                        break
                
                frame:np.ndarray = skimage.img_as_ubyte(frame)
                assert frame.ndim == 3
                assert frame.shape[2] == 3

                frame_bytes = frame.tobytes()
                for l in range(0, len(frame_bytes), self.pipe_chunksize):
                    r = min(l + self.pipe_chunksize, len(frame_bytes))
                    cp.stdin.write(frame_bytes[l:r])
            
            cp.stdin.flush()
            cp.stdin.close()

            cp.communicate()
            if cp.returncode != 0:
                raise RuntimeError('an error occurred while encoding using ffmpeg, consider setting `verbose=True` for more information')
            
            # now, open the new VideoIterator instance
            current_vid = VideoFrameIterator.from_temp_file(tempfile_path)

                
        return current_vid

    


if __name__ == '__main__':

    comp = H264Compression([H264CompressionParam(), H264CompressionParam(crf=10)])
    vit = VideoFrameIterator.from_preset_bigbuckbunny()
    new_vit = comp.transform(vit)

    for frame in new_vit:
        pass

