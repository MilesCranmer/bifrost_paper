from copy import deepcopy
 
import bifrost as bf
import bifrost.pipeline as bfp
from bifrost.ndarray import copy_array
 
 
class CopyBlock(bfp.TransformBlock):#                                         $\tikzmark{block-start}$
    """Copy the input ring to output ring"""
    def __init__(self, iring, space=None, *args, **kwargs):
        super(CopyBlock, self).__init__(iring, *args, **kwargs)
        if space is None:
            space = self.iring.space
        self.orings = [self.create_ring(space=space)]
    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        return ohdr
    def on_data(self, ispan, ospan):
        copy_array(ospan.data, ispan.data)#$\tikzmark{block-end}$
 
 

bc = bf.BlockChainer()



bc.blocks.read_wav(['hey_jude.wav'], gulp_nframe=4096) 
 
bc.blocks.copy(space='cuda')#                                $\tikzmark{gpu-start}$
bc.views.split_axis('time', 256, label='fine_time')
bc.blocks.fft(axes='fine_time', axis_labels='freq')
bc.blocks.detect(mode='scalar')
bc.blocks.transpose(['time', 'pol', 'freq'])#$\tikzmark{gpu-end}$
 
bc.blocks.copy(space='cuda_host') 
bc.blocks.quantize('i8')
bc.blocks.write_sigproc()
 
pipeline = bf.get_default_pipeline()# $\tikzmark{pipeline-start}$
pipeline.shutdown_on_signals() 
pipeline.run()#$\tikzmark{pipeline-end}$
 
