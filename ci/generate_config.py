import os
from glob import glob
import yaml

cfg = {
    "image": "yellow.hub.cambricon.com/magicmind/release/x86_64/magicmind:0.13.0-x86_64-ubuntu18.04-py_3_7",
    "variables":{
        "GIT_DEPTH": "1"
    },

    "build":{
        "stage":".pre",
        "tags":["mlu370-s4"],
        "script":[
            'rm -rf $(dirname "$(pwd)")/cnbox_resource',
            'ln -sf /workspace/cnbox_resource $(dirname "$(pwd)")/cnbox_resource',
            "ci/get_resource.sh",
            "python setup.py bdist_wheel"
        ],
        "artifacts":{
            "paths":["dist/*.whl"]
        }
    }
}

files = glob("./examples/*.sh")

for file in files:
    print(file)
    filename = os.path.split(file)[-1]
    name = filename.split(".")[0]
    cfg[name]= {
        "stage": "test",
        "tags":["mlu370-s4"],
        "script":[
            "pip install dist/*",
            'rm -rf $(dirname "$(pwd)")/cnbox_resource',
            'ln -sf /workspace/cnbox_resource $(dirname "$(pwd)")/cnbox_resource',
            file
        ]
    }

with open("jobs.yml", "w", encoding="utf-8") as f:
    yaml.dump(data=cfg, stream=f, allow_unicode=True)
