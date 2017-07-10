from copy import deepcopy
 
import bifrost as bf
from bifrost.pipeline import TransformBlock
from bifrost.ndarray import copy_array
 
 
class CopyBlock(TransformBlock):#                                         $\tikzmark{block-start}$
    """Copy the input ring to output ring"""
    def __init__(self, iring, space):
        super(CopyBlock, self).__init__(iring)
        self.orings = [self.create_ring(space=space)]
    def on_sequence(self, iseq):
        return deepcopy(iseq.header)
    def on_data(self, ispan, ospan):
        copy_array(ospan.data, ispan.data)#$\tikzmark{block-end}$
 
def copy_block(iring, space):
    return CopyBlock(iring, space)


bc = bf.BlockChainer()


bc.blocks.read_wav(['hey_jude.wav'], gulp_nframe=4096) 
 
bc.custom(copy_block)(space='cuda')#                                $\tikzmark{gpu-start}$
bc.views.split_axis('time', 256, label='fine_time')
bc.blocks.fft(axes='fine_time', axis_labels='freq')
bc.blocks.detect(mode='scalar')
bc.blocks.transpose(['time', 'pol', 'freq'])#$\tikzmark{gpu-end}$
 
bc.blocks.copy(space='system') 
bc.blocks.quantize('i8')
bc.blocks.write_sigproc()
 
pipeline = bf.get_default_pipeline()# $\tikzmark{pipeline-start}$
pipeline.shutdown_on_signals() 
pipeline.run()#$\tikzmark{pipeline-end}$
 
