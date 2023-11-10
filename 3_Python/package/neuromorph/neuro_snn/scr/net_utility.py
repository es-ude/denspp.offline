import os
from os.path import join, exists, abspath
from torch import save, load
from snntorch import utils


def save_net(net, path: str) -> None:
    path2model = abspath("../scr_ai")
    rel_path = join(path2model, path)
    utils.reset(net)
    if not exists(rel_path):
        os.makedirs(rel_path)
    if net.device != "cpu":
        net.module.cpu()
    model_path = "./"+path+net.get_name()+'.pt'
    save(net, model_path)


def load_net(path, file_name: str):
    path2model = os.path.abspath("../scr_ai")
    rel_path = os.path.join(path2model, path)
    com_path = os.path.join(rel_path, file_name+".pt")
    if not os.path.exists(com_path):
        raise Exception(f'model does not exist: {com_path}')
    else:
        path = os.path.join('../scr_ai', path)
        return load(os.path.join(path, file_name+".pt"))

