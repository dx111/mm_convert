from typing import List
import numpy as np
import os
import magicmind.python.runtime as mm

class MMInfer:
    def __init__(self, model_file) -> None:
        remote_ip = os.environ.get('REMOTE_IP', None)
        if remote_ip:
            sess = mm.remote.IRpcSession.rpc_connect(remote_ip + ":9009")
            self.dev = sess.get_device_by_id(0)
            self.dev.active()
            self.model = sess.create_model()
        else:
            self.dev = mm.Device()
            self.dev.id = 0
            assert self.dev.active().ok()
            self.model = mm.Model()
        self.queue = self.dev.create_queue()
        assert self.queue != None
        
        self.model.deserialize_from_file(model_file)    
        econfig = mm.Model.EngineConfig()
        econfig.device_type = "MLU"
        self.engine = self.model.create_i_engine(econfig)
        assert self.engine != None, "Failed to create engine"
        self.context = self.engine.create_i_context()
        self.inputs = self.context.create_inputs()
    
    def set_input_shape(self, i , shape):
        self.inputs[i].resize(shape)

    def predict(self, datas:List[np.ndarray] or np.ndarray) -> List[np.ndarray]:
        outputs = []
        # .shape = (1,18)
        # setattr(self.inputs[0], "shape", )
        if isinstance(datas, np.ndarray):
            datas = [datas]
        for i, data in enumerate(datas):
            if self.inputs[i].shape == tuple():
                continue
            if data.shape != self.inputs[i].shape:
                raise BaseException(f"shape not match, data shpae {data.shape}, network input shape {self.inputs[i].shape}")
            self.inputs[i].from_numpy(data)
            if isinstance(self.dev, mm.resources.Device):
                self.inputs[0].to(self.dev)
        status = self.context.enqueue(self.inputs, outputs, self.queue)
        if not status.ok():
            raise BaseException("enqueue failed")
  
        status = self.queue.sync()
        if not status.ok():
            raise BaseException("sync queue failed")
        return [x.asnumpy() for x in outputs]
