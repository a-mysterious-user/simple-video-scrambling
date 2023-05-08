from video_scramble.scrambler import BaseScrambler
from typing import List, Union, Tuple, Iterable
from collections import OrderedDict
import numpy as np

StepType = Tuple[str, BaseScrambler]

class ScramblePipeline:

    def __init__(self,
                 steps: Union[List[StepType], Tuple[StepType]],
                 ) -> None:
        

        self.named_steps = OrderedDict(steps)

        self.state = dict(
            timestamp=0
        )

        for step in self.named_steps.values():
            step.set_state(self.state)
        
    
    @staticmethod
    def _generic_transform(transform_steps:Iterable, 
                   transform_func_name:str, 
                   frame:np.ndarray, 
                   intermediate_steps:bool=False) -> Union[np.ndarray, OrderedDict]:
        
        if intermediate_steps:
            step_results = []

        cur_frame = frame
        for name, step in transform_steps:
            cur_frame = getattr(step, transform_func_name)(cur_frame)

            if intermediate_steps:
                step_results.append((name, cur_frame))

        if intermediate_steps:
            return OrderedDict(step_results)
        
        return cur_frame

    def increment_time(self) -> None:
        self.state['timestamp'] += 1

    def reset_time(self) -> None:
        self.state['timestamp']  = 0

    def fit(self, frame:np.ndarray) -> None:
        cur_frame = frame
        for step in self.named_steps.values():
            step.fit(cur_frame)
            cur_frame = step.transform(cur_frame)


    def transform(self, 
                  frame:np.ndarray, 
                  intermediate_steps:bool=False, 
                  increment_time:bool=True) -> Union[np.ndarray, OrderedDict]:
        ret = self._generic_transform(self.named_steps.items(), 'transform', frame, intermediate_steps=intermediate_steps)
        if increment_time:
            self.increment_time()
        return ret
    
    def inverse_transform(self, 
                          frame:np.ndarray, 
                          intermediate_steps:bool=False, 
                          increment_time:bool=True) -> Union[np.ndarray, OrderedDict]:
        ret = self._generic_transform(reversed(self.named_steps.items()), 
                                       'inverse_transform', frame, intermediate_steps=intermediate_steps)
        if increment_time:
            self.increment_time()
        return ret


if __name__ == '__main__':

    from video_scramble.video_handler import VideoFrameIterator
    from video_scramble.scrambler import BlockScramble, MultMod, BlockAdd, PixelRoll

    vid1 = VideoFrameIterator.from_preset_bigbuckbunny()

    vid_frames = list(vid1)

    steps = [
        #('multmod', MultMod()),
        ('blockscram', BlockScramble(patch_size=(128,128))),
        ('blockadd', BlockAdd(patch_size=(32,32))),
        ('roll', PixelRoll(roll_x=4, roll_y=4))
    ]
    ppl = ScramblePipeline(steps)
    ppl.fit(vid_frames[0])

    scrambled_frames = [ppl.transform(x) for x in vid_frames]

    vid_new = VideoFrameIterator.from_buf(scrambled_frames)
    vid_new.to_video('scrambled.mp4')

    ppl.reset_time()

    inv_scramble_frames = [ppl.inverse_transform(x) for x in scrambled_frames]

    vid_new = VideoFrameIterator.from_buf(inv_scramble_frames)
    vid_new.to_video('scrambled_inv.mp4')