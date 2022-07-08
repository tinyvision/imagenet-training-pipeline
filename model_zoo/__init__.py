import json

# move the `nas/models` folder in the LihgtNAS here 
from .models import __all_masternet__

_models = {
    'zennas' : __all_masternet__["MasterNet"],
}


def get_model(name, classes=1000):
        ### modify
    if ":" in name: 
        name_split = name.split(":")
        name = name_split[0]
        structure_txt = name_split[1]

    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)

    elif 'zennas' in name:
        if structure_txt:
            net = _models[name](num_classes=classes, 
                                structure_txt=structure_txt,
                                out_indices=(4,),
                                classfication=True)
        else:
            raise ValueError('zennas need structure file')
    else:
        net = _models[name](classes=classes)
    return net


def get_model_list():
    return _models.keys()
