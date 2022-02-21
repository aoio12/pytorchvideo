import os.path as osp
import shutil
import torch
import os


def save_checkpoint(state, is_best, filename, best_model_file, dir_data_name):
    file_path = osp.join(dir_data_name, filename)
    if not os.path.exists(dir_data_name):
        os.makedirs(dir_data_name)
    torch.save(state.state_dict(), file_path)
    if is_best:
        shutil.copyfile(file_path, osp.join(dir_data_name, best_model_file))
