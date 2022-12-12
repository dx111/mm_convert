import os
import pickle
import numpy as np
import magicmind.python.runtime as mm
from magicmind.python.common.types import get_datatype_by_numpy
from typing import List

HERE = os.path.split(os.path.abspath(__file__))[0]

def get_obj(idx, objs):
    if idx >= len(objs):
        obj = objs[0]
    else:
        obj = objs[idx]
    return obj

def print_error_and_exit(msg):
    print(msg)
    exit(1)

class CalibData(mm.CalibDataInterface):
    def __init__(self, data_list: List[np.ndarray]):
        super(CalibData, self).__init__()
        self._data_list = data_list
        self._cur_idx = 0
        status = self.reset()
        assert status.ok(), str(status)

    def next(self):
        if self._cur_idx >= len(self._data_list):
            return mm.Status(mm.Code.OUT_OF_RANGE, "No more data.")
        self._data = np.ascontiguousarray(self._data_list[self._cur_idx])
        self._data_shape = mm.Dims(self._data.shape)
        self._data_type = get_datatype_by_numpy(self._data.dtype)
        self._cur_idx += 1
        return mm.Status()

    def get_shape(self):
        return self._data_shape

    def get_data_type(self):
        return self._data_type

    def get_sample(self):
        return self._data

    def reset(self):
        self._cur_idx = 0
        return mm.Status()


class Calibrator(mm.Calibrator):
    def __init__(self, calibdata, algorithm):
        super(Calibrator, self).__init__(calibdata)
        status = self.set_quantization_algorithm(algorithm)
        assert status.ok(), str(status)

    def calibrate(self, network, builder_config=None):
        status = super(Calibrator, self).calibrate(network, builder_config)
        if not status.ok():
            return status
        return mm.Status()


class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}
    
    def register(self, target):
        def add_register_item(key, value):
            if not callable(value):
                raise Exception(f"register object must be callable! But receice:{value} is not callable!")
            if key in self._dict:
                print(f"warning: \033[33m{value.__name__} has been registered before, so we will overriden it\033[0m")
            self[key] = value
            return value

        if callable(target):
            return add_register_item(target.__name__, target)
        else:
            return lambda x : add_register_item(target, x)
    
    def __call__(self, target):
        return self.register(target)
    
    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]
    
    def __contains__(self, key):
        return key in self._dict
    
    def __str__(self):
        return str(self._dict)
    
    def keys(self):
        return self._dict.keys()
    
    def values(self):
        return self._dict.values()
    
    def items(self):
        return self._dict.items()

class Record:
    def __init__(self):
        self.data_dict = {}

    def add(self, *datas):
        for i, x in enumerate(datas):
            self.data_dict.setdefault(f"input{i}", []).append(x)
            
    def save(self, name = "calibrate_data"):
        pickle.dump(self.data_dict, open(name, "wb"))

