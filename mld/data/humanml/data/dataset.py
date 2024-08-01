import codecs as cs
import os
import random
from os.path import join as pjoin
import numpy as np
import spacy
import torch
from rich.progress import track
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from ..utils.get_opt import get_opt
from ..utils.word_vectorizer import WordVectorizer
from ..scripts.motion_process import recover_root_rot_pos
from ..common.quaternion import quaternion_to_cont6d


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


# import spacy
def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def build_lable_list(filename):
    label_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                npy_name, _, label = parts
                if label not in label_dict:
                    label_dict[label] = []
                label_dict[label].append(npy_name)
    return label_dict

def build_dict_from_txt(filename,is_style=True,is_style_text=False):
    result_dict = {}
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) >= 3:
                key = parts[0]
                if is_style and is_style_text == False:
                    value = parts[2]
                elif is_style_text:
                    value = parts[1].split("_")[0]
                else:
                    value = parts[3]


                result_dict[key] = value
                
    return result_dict


def build_dict_from_style_to_neutral(filename):
    result_dict = {}
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) >= 3:
                key = parts[0]
                value = parts[3]
                result_dict[key] = value
                
    return result_dict
"""For use of training style label classify model"""

def shuffle_segments_numpy(data, segment_length):
    if data.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")

    T, H = data.shape

    num_segments = T // segment_length

    shuffled_array = data.copy()

    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        shuffled_indices = np.random.permutation(segment_length) + start
        shuffled_array[start:end, :] = data[shuffled_indices, :]

    return shuffled_array


def random_zero_out(data, percentage=0.4, probability=0.6,noise_probability=0.8,noise_level=0.05):
    if random.random() < probability:
        # Calculate the total number of sequences to zero out
        num_sequences = data.shape[0]

        percentage = np.random.rand() * 0.5
        num_to_zero_out = int(num_sequences * percentage)

        # Randomly choose sequence indices to zero out
        indices_to_zero_out = np.random.choice(num_sequences, num_to_zero_out, replace=False)

        # Zero out the chosen sequences
        data[indices_to_zero_out, :] = 0

    # data = shuffle_segments_numpy(data,16)

    if random.random() < noise_probability:
        noise = np.random.normal(0, noise_level * np.ptp(data), data.shape)
        data += noise

    return data

class MotionDatasetPuzzle(data.Dataset):
    def __init__(
        self,stage="train"
    ):
        # self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = 196
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        self.min_motion_length = 40
        # self.max_text_len = max_text_len
        self.unit_length = 4

        data_dict_1 = {}
        id_list_1 = []

        data_dict_2 = {}
        id_list_2 = []

        split_dir = "/work/vig/zhonglei/stylized_motion/dataset_all/"
        motion_dir = "/work/vig/zhonglei/stylized_motion/dataset_all/new_joint_vecs"
        mean = np.load("/work/vig/zhonglei/stylized_motion/dataset_all/Mean.npy")
        std = np.load("/work/vig/zhonglei/stylized_motion/dataset_all/Std.npy")

        # split_dir = "/work/vig/zhonglei/stylized_motion/dataset/mocap"
        # motion_dir = "/work/vig/zhonglei/stylized_motion/dataset/mocap/new_joint_vecs"
        # mean = np.load("/work/vig/zhonglei/stylized_motion/dataset/mocap/Mean.npy")
        # std = np.load("/work/vig/zhonglei/stylized_motion/dataset/mocap/Std.npy")

        # split_dir = os.path.dirname(split_file)
        # split_base = os.path.basename(split_file).split(".")[0]
        split_subfile_1 = os.path.join(split_dir,stage + ".txt")#_humanml
        split_subfile_2 = os.path.join(split_dir,stage + ".txt")#_100STYLE

        with cs.open(split_subfile_1, "r") as f:
            for line in f.readlines():
                id_list_1.append(line.strip())
        self.id_list_1 = id_list_1

        with cs.open(split_subfile_2, "r") as f:
            for line in f.readlines():
                id_list_2.append(line.strip())
        self.id_list_2 = id_list_2

        progress_bar = True
        maxdata = 1e10

        if progress_bar:
            enumerator_1 = enumerate(
                track(
                    id_list_1,
                    f"Loading 100STYLE {split_subfile_1.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator_1 = enumerate(id_list_1)

        count = 0
        bad_count = 0
        new_name_list_1 = []
        length_list_1 = []

        for i, name in enumerator_1:
            if count > maxdata:
                break
            motion = np.load(pjoin(motion_dir, name + ".npy"))
            if (len(motion)) < self.min_motion_length or (len(motion) >=200):
                bad_count += 1
                continue
            text_data_1 = []
            flag = True

            if flag:
                data_dict_1[name] = {
                    "motion": motion,
                    "length": len(motion),
                        # "text": text_data_1,
                }
                new_name_list_1.append(name)
                length_list_1.append(len(motion))
                count += 1 
                           

        name_list_1, length_list_1 = zip(
            *sorted(zip(new_name_list_1, length_list_1), key=lambda x: x[1]))

        if progress_bar:
            enumerator_2 = enumerate(
                track(
                    id_list_2,
                    f"Loading HumanML3D {split_subfile_2.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator_2 = enumerate(id_list_2)

        count = 0
        bad_count = 0
        new_name_list_2 = []
        length_list_2 = []

        for i, name in enumerator_2:
            if count > maxdata:
                break
            motion = np.load(pjoin(motion_dir, name + ".npy"))
            if (len(motion)) < self.min_motion_length or (len(motion) >=200):
                bad_count += 1
                continue

            if flag:
                data_dict_2[name] = {
                    "motion": motion,
                    "length": len(motion),
                        # "text": text_data_2,
                }
                new_name_list_2.append(name)
                length_list_2.append(len(motion))
                count += 1            

        name_list_2, length_list_2 = zip(
            *sorted(zip(new_name_list_2, length_list_2), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr_1 = np.array(length_list_1)
        self.data_dict_1 = data_dict_1
        self.name_list_1 = name_list_1

        self.length_arr_2 = np.array(length_list_2)
        self.data_dict_2 = data_dict_2
        self.name_list_2 = name_list_2

        self.nfeats = motion.shape[1]
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length

        self.pointer_1 = np.searchsorted(self.length_arr_1, length)
        print("Pointer Pointing at %d" % self.pointer_1)

        # self.pointer_2 = np.searchsorted(self.length_arr_2, length)
        # print("Pointer Pointing at %d" % self.pointer_2)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def transform(self,data):
        return (data - self.mean) / self.std

    def get_mean_std(self):
        return self.mean, self.std
    
    def __len__(self):
        return len(self.name_list_1) - self.pointer
    
    def recover_from_ric2(self,data, joints_num):
        r_rot_quat, r_pos = recover_root_rot_pos(data)

        positions = data[..., 4:(joints_num - 1) * 3 + 4]
        positions = positions.view(positions.shape[:-1] + (-1, 3))

        '''Add Y-axis rotation to local joints'''
        # positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

        '''Add root XZ to joints'''
        # positions[..., 0] += r_pos[..., 0:1]
        # positions[..., 2] += r_pos[..., 2:3]

        '''Concate root and joints'''
        positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

        return positions.numpy()

    def recover_rot(self,data):
        # dataset [bs, seqlen, 263/251] HumanML/KIT
        joints_num = 22 if data.shape[-1] == 263 else 21
        r_rot_quat, r_pos = recover_root_rot_pos(data)
        r_pos_pad = torch.cat([r_pos, torch.zeros_like(r_pos)], dim=-1).unsqueeze(-2)
        r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)
        start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
        end_indx = start_indx + (joints_num - 1) * 6
        cont6d_params = data[..., start_indx:end_indx]
        cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
        cont6d_params = cont6d_params.view(-1, joints_num, 6)
        cont6d_params = torch.cat([cont6d_params, r_pos_pad], dim=-2)
        return cont6d_params.numpy()

    def extract_tensor(self,motion):
        cont6d_params = self.recover_rot(torch.from_numpy(motion).float())
        positions = self.recover_from_ric2(torch.from_numpy(motion).float(),22)

        # print("motion",motion.shape)
        # print("positions",positions.shape)
        # print("cont6d_params",cont6d_params[:,:-1].shape)
        root_tensor = motion[:,0:4]
        feet =motion[:,-4:]

        ric_data = motion[:,4:4 + 21*3].reshape(self.max_motion_length,21,-1)
        rot_data = motion[:,4 + 21*3 : 4 + 21*3 + 21*6].reshape(self.max_motion_length,21,-1)
        local_vel = motion[:,4 + 21*3 + 21*6 : 4 + 21*3 + 21*6 + 22*3].reshape(self.max_motion_length,22,-1)
        root_vel = local_vel[:,0:1,:]

        # print("rot_data",rot_data.shape)
        # motion_ = np.concatenate([ric_data,rot_data,local_vel[:,1:,:]],axis=2)
        motion_ = np.concatenate([positions,cont6d_params[:,:-1],local_vel],axis=2)

        # print("motion_",motion_.shape)
        # print("motion_2",motion_2.shape)
        return root_tensor,motion_,root_vel,feet

    def __getitem__(self, item):
        idx_1 = (self.pointer_1 + item) #% len(self.name_list_1)
        data_1 = self.data_dict_1[self.name_list_1[idx_1]]
        motion_1, m_length_1 = data_1["motion"], data_1["length"]

        idx_2 = (self.pointer_1 + item + random.randint(0,len(self.name_list_2)-1)) % len(self.name_list_2)
        data_2 = self.data_dict_2[self.name_list_2[idx_2]]
        motion_2, m_length_2 = data_2["motion"], data_2["length"]
      
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length_1 = (m_length_1 // self.unit_length - 1) * self.unit_length
            m_length_2 = (m_length_2 // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length_1 = (m_length_1 // self.unit_length) * self.unit_length
            m_length_2 = (m_length_2 // self.unit_length) * self.unit_length
        
        idx_1 = random.randint(0, len(motion_1) - m_length_1)
        motion_1 = motion_1[idx_1:idx_1 + m_length_1]
        "Z Normalization"
        motion_1 = (motion_1 - self.mean) / self.std

        idx_2 = random.randint(0, len(motion_2) - m_length_2)
        motion_2 = motion_2[idx_2:idx_2 + m_length_2]
        "Z Normalization"
        motion_2 = (motion_2 - self.mean) / self.std

        # debug check nan
        if np.any(np.isnan(motion_1)):
            raise ValueError("nan in motion")
        
        if m_length_1 < self.max_motion_length:
            motion_1 = np.concatenate([motion_1,
                                     np.zeros((self.max_motion_length - m_length_1, motion_1.shape[1]))
                                     ], axis=0)

        if m_length_2 < self.max_motion_length:
            motion_2 = np.concatenate([motion_2,
                                     np.zeros((self.max_motion_length - m_length_2, motion_2.shape[1]))
                                     ], axis=0)

        #batch [seq,263]

        root_tensor_1,motion_1_,root_vel_1,feet_1 = self.extract_tensor(motion_1)
        root_tensor_2,motion_2_,root_vel_2,feet_2 = self.extract_tensor(motion_2)

        return {
            "motion_1": torch.from_numpy(motion_1).cuda(),
            "root_tensor_1": torch.from_numpy(root_tensor_1).cuda(),
            "root_vel_1": torch.from_numpy(root_vel_1).cuda(),
            "feet_1":torch.from_numpy(feet_1).cuda(),

            "motion_2": torch.from_numpy(motion_2).cuda(),
            "root_tensor_2": torch.from_numpy(root_tensor_2).cuda(),
            "root_vel_2": torch.from_numpy(root_vel_2).cuda(),
            "feet_2":torch.from_numpy(feet_2).cuda(),
        }


class StyleMotionDataset(data.Dataset):

    def __init__(self, stage = 'train'):
        data_dict = {}
        id_list = []
        self.max_length = 20
        self.max_motion_length = 196
        self.unit_length = 4
        self.pointer = 0

        txt_path = "/work/vig/zhonglei/stylized_motion/dataset_all/"
        # txt_path = "/work/vig/zhonglei/stylized_motion/dataset/mocap/"
        if stage == 'train':
            # split_file = txt_path + "train_100STYLE_Full.txt"
            split_file = txt_path + "train_100STYLE_Filter.txt"
        elif stage == 'test':
            split_file = txt_path + "test_100STYLE_Filter.txt" #test_100STYLE

        self.stage = stage
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # dict_path = "/work/vig/zhonglei/stylized_motion/dataset/mocap/mocap_name_dict.txt"
        dict_path = "/work/vig/zhonglei/stylized_motion/dataset_all/100STYLE_name_dict.txt"
        motion_to_label = build_dict_from_txt(dict_path)
        
        # mean = np.load("/work/vig/zhonglei/stylized_motion/dataset/mocap/Mean.npy")
        # std = np.load("/work/vig/zhonglei/stylized_motion/dataset/mocap/Std.npy")
        mean = np.load("/work/vig/zhonglei/stylized_motion/dataset_all/Mean.npy")
        std = np.load("/work/vig/zhonglei/stylized_motion/dataset_all/Std.npy")
        new_name_list = []
        length_list = []
        label_list = []
        count = 0
        bad_count = 0
        new_name_list = []

        # motion_dir = "/work/vig/zhonglei/stylized_motion/dataset/mocap/new_joint_vecs"
        motion_dir = "/work/vig/zhonglei/stylized_motion/dataset_all/new_joint_vecs"
        text_dir = "/work/vig/zhonglei/stylized_motion/dataset_all/texts"
        length_list = []

        # id_list = id_list[:500]
        enumerator = enumerate(
            track(
                id_list,
                f"Loading 100STYLE {split_file.split('/')[-1].split('.')[0]}",
            ))
        maxdata = 1e10
        self.min_motion_length = 40

        for i, name in enumerator:
            if count > maxdata:
                break

            motion = np.load(pjoin(motion_dir, name + ".npy"))
            label_data = motion_to_label[name]
            # label_content_data = motion_to_content_label[name]

            if (len(motion)) < self.min_motion_length:
                continue
            text_data = []
            
            text_path = pjoin(text_dir, name + ".txt")
            assert os.path.exists(text_path)
            with cs.open(text_path) as f:
                for line in f.readlines():
                    text_dict_2 = {}
                    line_split = line.strip().split("#")
                    caption = line_split[0]
                    text_dict_2["caption"] = caption
                    text_data.append(text_dict_2)
            
            data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "label": label_data,
                        "text": text_data,
                    }
 
            new_name_list.append(name)
            length_list.append(len(motion))
            count += 1
        
        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        # self.reset_max_len(self.max_length)
    
    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def transform(self,data):
        return (data - self.mean) / self.std

    def get_mean_std(self):
        return self.mean, self.std
    
    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, label, text_list = data["motion"], data["length"], data["label"], data["text"]

        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        

        text_data = random.choice(text_list)
        caption = text_data["caption"]


        m_length = min(196,m_length)
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if self.stage == 'train':
            motion = random_zero_out(motion)

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        return {
            "motion": torch.from_numpy(motion).cuda(),
            "label":torch.tensor(int(label)).cuda(),
            "length":m_length,
            "text":caption
        }


class ContentMotionDataset(data.Dataset):

    def __init__(self, stage = 'train'):
        data_dict = {}
        id_list = []
        self.max_length = 20
        self.max_motion_length = 196
        self.unit_length = 4
        self.pointer = 0

        # txt_path = "/work/vig/zhonglei/stylized_motion/dataset_all/"
        txt_path = "/work/vig/zhonglei/stylized_motion/dataset/mocap/"
        if stage == 'train':
            # split_file = txt_path + "train_100STYLE_Full.txt"
            split_file = txt_path + "train.txt"
        elif stage == 'test':
            split_file = txt_path + "test.txt" #test_100STYLE

        self.stage = stage
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        dict_path = "/work/vig/zhonglei/stylized_motion/dataset/mocap/mocap_name_dict_content.txt"
        motion_to_label = build_dict_from_txt(dict_path,is_style = False)

        mean = np.load("/work/vig/zhonglei/stylized_motion/dataset/mocap/Mean.npy")
        std = np.load("/work/vig/zhonglei/stylized_motion/dataset/mocap/Std.npy")

        new_name_list = []
        length_list = []
        label_list = []
        count = 0
        bad_count = 0
        new_name_list = []

        motion_dir = "/work/vig/zhonglei/stylized_motion/dataset/mocap/new_joint_vecs"
        text_dir = "/work/vig/zhonglei/stylized_motion/dataset/mocap/texts"
        length_list = []

        # id_list = id_list[:500]
        enumerator = enumerate(
            track(
                id_list,
                f"Loading mocap xia {split_file.split('/')[-1].split('.')[0]}",
            ))
        maxdata = 1e10
        self.min_motion_length = 40

        for i, name in enumerator:
            if count > maxdata:
                break

            motion = np.load(pjoin(motion_dir, name + ".npy"))
            label_data = motion_to_label[name]

            if (len(motion)) < self.min_motion_length:
                continue
            text_data = []
            
            text_path = pjoin(text_dir, name + ".txt")
            assert os.path.exists(text_path)
            with cs.open(text_path) as f:
                for line in f.readlines():
                    text_dict_2 = {}
                    line_split = line.strip().split("#")
                    caption = line_split[0]
                    text_dict_2["caption"] = caption
                    text_data.append(text_dict_2)
            
            data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "label": label_data,
                        "text": text_data,
                    }
 
            new_name_list.append(name)
            length_list.append(len(motion))
            count += 1
        
        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        # self.reset_max_len(self.max_length)
    
    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def transform(self,data):
        return (data - self.mean) / self.std

    def get_mean_std(self):
        return self.mean, self.std
    
    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, label,text_list = data["motion"], data["length"], data["label"], data["text"]

        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        

        text_data = random.choice(text_list)
        caption = text_data["caption"]


        m_length = min(196,m_length)
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if self.stage == 'train':
            motion = random_zero_out(motion)

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        return {
            "motion": torch.from_numpy(motion).cuda(),
            "label":torch.tensor(int(label)).cuda(),
            "length":m_length,
            "text":caption
        }


class StyleMotionDatasetTri(data.Dataset):

    def __init__(self, stage = 'train'):
        data_dict = {}
        id_list = []
        self.max_length = 20
        self.max_motion_length = 196
        self.unit_length = 4
        self.pointer = 0

        txt_path = "/work/vig/zhonglei/stylized_motion/dataset_all/"
        # txt_path = "/work/vig/zhonglei/stylized_motion/dataset/mocap/"
        if stage == 'train':
            split_file = txt_path + "train_100STYLE_Full.txt"
        elif stage == 'test':
            split_file = txt_path + "test_100STYLE_Full.txt" #test_100STYLE

        self.stage = stage
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # dict_path = "/work/vig/zhonglei/stylized_motion/dataset/mocap/mocap_name_dict.txt"
        dict_path = "/work/vig/zhonglei/stylized_motion/dataset_all/100STYLE_name_dict.txt"
        motion_to_label = build_dict_from_txt(dict_path)
        self.label_to_list = build_lable_list(dict_path)

        

        # mean = np.load("/work/vig/zhonglei/stylized_motion/dataset/mocap/Mean.npy")
        # std = np.load("/work/vig/zhonglei/stylized_motion/dataset/mocap/Std.npy")
        mean = np.load("/work/vig/zhonglei/stylized_motion/dataset_all/Mean.npy")
        std = np.load("/work/vig/zhonglei/stylized_motion/dataset_all/Std.npy")
        new_name_list = []
        length_list = []
        label_list = []
        count = 0
        bad_count = 0
        new_name_list = []

        # motion_dir = "/work/vig/zhonglei/stylized_motion/dataset/mocap/new_joint_vecs"
        motion_dir = "/work/vig/zhonglei/stylized_motion/dataset_all/new_joint_vecs"
        length_list = []

        enumerator = enumerate(
            track(
                id_list,
                f"Loading 100STYLE {split_file.split('/')[-1].split('.')[0]}",
            ))
        maxdata = 1e10
        self.min_motion_length = 40
        for i, name in enumerator:
            if count > maxdata:
                break

            motion = np.load(pjoin(motion_dir, name + ".npy"))
            label_data = motion_to_label[name]
            if (len(motion)) < self.min_motion_length:
                continue
            data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "label": label_data,
                    }
 
            new_name_list.append(name)
            length_list.append(len(motion))
            count += 1
        
        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        # self.reset_max_len(self.max_length)
    
    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def transform(self,data):
        return (data - self.mean) / self.std

    def get_mean_std(self):
        return self.mean, self.std
    
    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, label = data["motion"], data["length"], data["label"]

        label_list = self.label_to_list[label]
        list_len = len(label_list)
        
        random_name = None
        while True:
            random_index = random.randint(0, list_len - 1)
            if random_index != idx:
                random_name = self.name_list[random_index]
                break

        data_1_same = self.data_dict[random_name]
        motion_1_same, m_length_1_same = data_1_same["motion"], data_1_same["length"]

        label_list_neg = self.label_to_list[str((int(label) + 1) % 100)]
        list_len_neg = len(label_list_neg)
        
        random_name_neg = self.name_list[random.randint(0, list_len_neg - 1)]
        data_1_neg = self.data_dict[random_name_neg]
        motion_1_neg, m_length_1_neg = data_1_neg["motion"], data_1_neg["length"]

        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
            m_length_1_same = (m_length_1_same // self.unit_length - 1) * self.unit_length
            m_length_1_neg = (m_length_1_neg // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
            m_length_1_same = (m_length_1_same // self.unit_length) * self.unit_length
            m_length_1_neg = (m_length_1_neg // self.unit_length) * self.unit_length
        
        m_length = min(196,m_length)
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        m_length_1_same = min(196,m_length_1_same)
        idx_1_same = random.randint(0, len(motion_1_same) - m_length_1_same)
        motion_1_same = motion_1_same[idx_1_same:idx_1_same + m_length_1_same]

        m_length_1_neg = min(196,m_length_1_neg)
        idx_1_neg = random.randint(0, len(motion_1_neg) - m_length_1_neg)
        motion_1_neg = motion_1_neg[idx_1_neg:idx_1_neg + m_length_1_neg]
        
        "Z Normalization"
        motion = (motion - self.mean) / self.std
        motion_1_same = (motion_1_same - self.mean) / self.std
        motion_1_neg = (motion_1_neg - self.mean) / self.std

        if self.stage == 'train':
            motion = random_zero_out(motion)
            motion_1_same = random_zero_out(motion_1_same)
            motion_1_neg = random_zero_out(motion_1_neg)

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,np.zeros((self.max_motion_length - m_length, motion.shape[1]))], axis=0)
        
        if m_length_1_same < self.max_motion_length:
            motion_1_same = np.concatenate([motion_1_same,np.zeros((self.max_motion_length - m_length_1_same, motion_1_same.shape[1]))], axis=0)
        
        if m_length_1_neg < self.max_motion_length:
            motion_1_neg = np.concatenate([motion_1_neg,np.zeros((self.max_motion_length - m_length_1_neg, motion_1_neg.shape[1]))], axis=0)

        return {
            "motion": torch.from_numpy(motion).cuda(),
            "motion_pos": torch.from_numpy(motion_1_same).cuda(),
            "motion_neg": torch.from_numpy(motion_1_neg).cuda(),
            "label":torch.tensor(int(label)).cuda()
        }
"""For use of training text-2-motion generative model"""
class Text2MotionDataset(data.Dataset):

    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        min_motion_len = 40 if self.opt.dataset_name == "t2m" else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(n_motion)) < min_motion_len or (
                                        len(n_motion) >= 200):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4:4 + (joints_num - 1) * 3] = std[4:4 +
                                                  (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3:4 +
                (joints_num - 1) * 9] = (std[4 + (joints_num - 1) * 3:4 +
                                             (joints_num - 1) * 9] / 1.0)
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9:4 + (joints_num - 1) * 9 +
                joints_num * 3] = (std[4 + (joints_num - 1) * 9:4 +
                                       (joints_num - 1) * 9 + joints_num * 3] /
                                   1.0)
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = (
                std[4 +
                    (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias)

            assert 4 + (joints_num -
                        1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, "mean.npy"), mean)
            np.save(pjoin(opt.meta_dir, "std.npy"), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.opt.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if self.opt.is_train:
            if m_length != self.max_length:
                # print("Motion original length:%d_%d"%(m_length, len(motion)))
                if self.opt.unit_length < 10:
                    coin2 = np.random.choice(["single", "single", "double"])
                else:
                    coin2 = "single"
                if len_gap == 0 or (len_gap == 1 and coin2 == "double"):
                    m_length = self.max_length
                    idx = random.randint(0, m_length - self.max_length)
                    motion = motion[idx:idx + self.max_length]
                else:
                    if coin2 == "single":
                        n_m_length = self.max_length + self.opt.unit_length * len_gap
                    else:
                        n_m_length = self.max_length + self.opt.unit_length * (
                            len_gap - 1)
                    idx = random.randint(0, m_length - n_m_length)
                    motion = motion[idx:idx + self.max_length]
                    m_length = n_m_length
                # print(len_gap, idx, coin2)
        else:
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(["single", "single", "double"])
            else:
                coin2 = "single"

            if coin2 == "double":
                m_length = (m_length // self.opt.unit_length -
                            1) * self.opt.unit_length
            elif coin2 == "single":
                m_length = (m_length //
                            self.opt.unit_length) * self.opt.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx + m_length]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length

class Text2MotionDatasetCMLDTest(data.Dataset):

    def __init__(
        self,
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        text_dir,
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict_1 = {}
        id_list_1 = []

        data_dict_2 = {}
        id_list_2 = []

        split_dir = os.path.dirname(split_file)
        split_base = os.path.basename(split_file).split(".")[0]
        split_subfile_1 = os.path.join(split_dir,split_base + "_humanml.txt")#_humanml
        split_subfile_2 = os.path.join(split_dir,split_base + "_100STYLE_Filter.txt")#_100STYLE_Filter

        dict_path = "/work/vig/zhonglei/stylized_motion/dataset_all/100STYLE_name_dict_Filter.txt"
        motion_to_label = build_dict_from_txt(dict_path)
        motion_to_style_text = build_dict_from_txt(dict_path,is_style_text=True)

        with cs.open(split_subfile_1, "r") as f:
            for line in f.readlines():
                id_list_1.append(line.strip())
        
        num = len(id_list_1)
        
        random_samples = np.random.choice(range(num), size=100, replace=False)
        id_list_1 = np.array(id_list_1)
        # Use random_samples to index id_list_1_np
        # id_list_1 = id_list_1[random_samples]

        self.id_list_1 = id_list_1

        with cs.open(split_subfile_2, "r") as f:
            for line in f.readlines():
                id_list_2.append(line.strip())
        
        id_list_2 = id_list_2
        self.id_list_2 = id_list_2

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        if progress_bar:
            enumerator_1 = enumerate(
                track(
                    id_list_1,
                    f"Loading HumanML3D {split_subfile_1.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator_1 = enumerate(id_list_1)

        count = 0
        bad_count = 0
        new_name_list_1 = []
        length_list_1 = []

        for i, name in enumerator_1:
            if count > maxdata:
                break
            motion = np.load(pjoin(motion_dir, name + ".npy"))
            if (len(motion)) < self.min_motion_length or (len(motion) >=200):
                bad_count += 1
                continue
            text_data_1 = []
            flag = False

            # name = style_to_neutral[name]
            text_path = pjoin(text_dir, name + ".txt")
            assert os.path.exists(text_path)
            with cs.open(text_path) as f:
                for line in f.readlines():
                    text_dict_1 = {}
                    line_split = line.strip().split("#")

                    caption = line_split[0]
                    tokens = line_split[1].split(" ")
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict_1["caption"] = caption
                    text_dict_1["tokens"] = tokens
                    text_data_1.append(text_dict_1)
                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data_1.append(text_dict_1)
                    else:
                        try:
                            n_motion = motion[int(f_tag * 20):int(to_tag *20)]
                            if (len(n_motion)) < self.min_motion_length or ((len(n_motion) >= 200)):
                                continue
                            new_name = (
                                random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                "_" + name)
                            while new_name in data_dict_1:
                                new_name = (random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" +name)
                            data_dict_1[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict_1],
                                }
                            new_name_list_1.append(new_name)
                            length_list_1.append(len(n_motion))
                        except:
                            print(line_split)
                            print(line_split[2], line_split[3], f_tag,to_tag, name)

                
                if flag:
                    data_dict_1[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data_1,
                    }
                    new_name_list_1.append(name)
                    length_list_1.append(len(motion))
                    count += 1            

        name_list_1, length_list_1 = zip(
            *sorted(zip(new_name_list_1, length_list_1), key=lambda x: x[1]))

        if progress_bar:
            enumerator_2 = enumerate(
                track(
                    id_list_2,
                    f"Loading 100STYLE {split_subfile_2.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator_2 = enumerate(id_list_2)

        count = 0
        bad_count = 0
        new_name_list_2 = []
        length_list_2 = []

        for i, name in enumerator_2:
            if count > maxdata:
                break
            motion = np.load(pjoin(motion_dir, name + ".npy"))
            label_data = motion_to_label[name]
            style_text = motion_to_style_text[name]

            if (len(motion)) < self.min_motion_length or (len(motion) >=200):
                bad_count += 1
                continue
            text_data_2 = []
            flag = True

            text_path = pjoin(text_dir, name + ".txt")
            assert os.path.exists(text_path)
            with cs.open(text_path) as f:
                for line in f.readlines():
                    text_dict_2 = {}
                    line_split = line.strip().split("#")
                    caption = line_split[0]
                    text_dict_2["caption"] = caption
                    text_data_2.append(text_dict_2)

                if flag:
                    data_dict_2[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data_2,
                        "label": label_data,
                        "style_text":style_text,
                    }
                    new_name_list_2.append(name)
                    length_list_2.append(len(motion))
                    count += 1            

        name_list_2, length_list_2 = zip(
            *sorted(zip(new_name_list_2, length_list_2), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr_1 = np.array(length_list_1)
        self.data_dict_1 = data_dict_1
        self.name_list = name_list_1

        self.length_arr_2 = np.array(length_list_2)
        self.data_dict_2 = data_dict_2
        self.name_list_2 = name_list_2

        self.nfeats = motion.shape[1]
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length

        self.pointer_1 = np.searchsorted(self.length_arr_1, length)
        print("Pointer Pointing at %d" % self.pointer_1)

        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def transform(self,data):
        return (data - self.mean) / self.std

    def get_mean_std(self):
        return self.mean, self.std
    
    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx_1 = (self.pointer_1 + item) % len(self.name_list_1)
        data_1 = self.data_dict_1[self.name_list[idx_1]]
        motion_1, m_length_1, text_list_1 = data_1["motion"], data_1["length"], data_1["text"]

        idx_2 = (self.pointer_1 + item + random.randint(0,len(self.name_list_2)-1)) % len(self.name_list_2)
        data_2 = self.data_dict_2[self.name_list_2[idx_2]]
        motion_2, m_length_2, text_list_2,label,style_text = data_2["motion"], data_2["length"], data_2["text"], data_2["label"], data_2["style_text"]
      
        # Randomly select a caption
        text_data_1 = random.choice(text_list_1)
        caption_1,tokens = text_data_1["caption"], text_data_1["tokens"]

        text_data_2 = random.choice(text_list_2)
        caption_2 = text_data_2["caption"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)


        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length_1 = (m_length_1 // self.unit_length - 1) * self.unit_length
            m_length_2 = (m_length_2 // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length_1 = (m_length_1 // self.unit_length) * self.unit_length
            m_length_2 = (m_length_2 // self.unit_length) * self.unit_length
        
        idx_1 = random.randint(0, len(motion_1) - m_length_1)
        motion_1 = motion_1[idx_1:idx_1 + m_length_1]
        "Z Normalization"
        motion_1 = (motion_1 - self.mean) / self.std

        idx_2 = random.randint(0, len(motion_2) - m_length_2)
        motion_2 = motion_2[idx_2:idx_2 + m_length_2]
        "Z Normalization"
        motion_2 = (motion_2 - self.mean) / self.std

        # debug check nan
        if np.any(np.isnan(motion_1)):
            raise ValueError("nan in motion")

        return (
            word_embeddings,
            pos_one_hots,
            caption_1,
            sent_len,
            motion_1,
            m_length_1,
            "_".join(tokens),

            caption_2,
            motion_2,
            m_length_2,
            label,
            style_text,
        )


"""For use of training text motion matching model, and evaluations"""
class Text2MotionDatasetV2(data.Dataset):

    def __init__(
        self,
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        text_dir,
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
            
        self.id_list = id_list

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading HumanML3D {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)
        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            motion = np.load(pjoin(motion_dir, name + ".npy"))
            if (len(motion)) < self.min_motion_length or (len(motion) >=200):
                bad_count += 1
                continue
            text_data = []
            flag = False

            text_path = pjoin(text_dir, name + ".txt")
            text_path_template = pjoin(text_dir, "000000" + ".txt")
            # print("text_path")
            if not os.path.exists(text_path):
                text_path = text_path_template
                print("no exist!!!!")
            with cs.open(text_path) as f:
                for line in f.readlines():
                    text_dict = {}
                    line_split = line.strip().split("#")
                    caption = line_split[0]
                    tokens = line_split[1].split(" ")
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict["caption"] = caption
                    text_dict["tokens"] = tokens
                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                    else:
                        try:
                            n_motion = motion[int(f_tag * 20):int(to_tag *20)]
                            if (len(n_motion)) < self.min_motion_length or ((len(n_motion) >= 200)):
                                continue
                            new_name = (
                                random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                "_" + name)
                            while new_name in data_dict:
                                new_name = (random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" +name)
                            data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                            new_name_list.append(new_name)
                            length_list.append(len(n_motion))
                        except:
                            print(line_split)
                            print(line_split[2], line_split[3], f_tag,to_tag, name)

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    # print(count)
                    count += 1
                    # print(name)
            

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def transform(self,data):
        return (data - self.mean) / self.std

    def get_mean_std(self):
        return self.mean, self.std
    
    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        "Z Normalization"
        motion = (motion - self.mean) / self.std


        # debug check nan
        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
        )
        # return caption, motion, m_length


class Text2MotionDatasetCMLD(data.Dataset):
    def __init__(
        self,
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        text_dir,
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict_1 = {}
        id_list_1 = []

        data_dict_2 = {}
        id_list_2 = []

        split_dir = os.path.dirname(split_file)
        split_base = os.path.basename(split_file).split(".")[0]
        split_subfile_1 = os.path.join(split_dir,split_base + "_100STYLE_Full.txt")#_humanml
        # split_subfile_1 = os.path.join(split_dir,split_base + ".txt")#_humanml
        split_subfile_2 = os.path.join(split_dir,split_base + "_humanml.txt")#_100STYLE

        # split_subfile_1 = os.path.join(split_dir,split_base + "_mocap.txt")#_humanml
        # split_subfile_2 = os.path.join(split_dir,split_base + "_humanml.txt")#_100STYLE

        with cs.open(split_subfile_1, "r") as f:
            for line in f.readlines():
                id_list_1.append(line.strip())
        self.id_list_1 = id_list_1

        with cs.open(split_subfile_2, "r") as f:
            for line in f.readlines():
                id_list_2.append(line.strip())
        self.id_list_2 = id_list_2

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        if progress_bar:
            enumerator_1 = enumerate(
                track(
                    id_list_1,
                    f"Loading 100STYLE {split_subfile_1.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator_1 = enumerate(id_list_1)

        count = 0
        bad_count = 0
        new_name_list_1 = []
        length_list_1 = []

        for i, name in enumerator_1:
            if count > maxdata:
                break
            motion = np.load(pjoin(motion_dir, name + ".npy"))
            if (len(motion)) < self.min_motion_length or (len(motion) >=200):
                bad_count += 1
                continue
            text_data_1 = []
            flag = True

            # name = style_to_neutral[name]
            text_path = pjoin(text_dir, name + ".txt")
            assert os.path.exists(text_path)
            with cs.open(text_path) as f:
                for line in f.readlines():
                    text_dict_1 = {}
                    line_split = line.strip().split("#")
                    caption = line_split[0]
                    text_dict_1["caption"] = caption
                    text_data_1.append(text_dict_1)
                
                if flag:
                    data_dict_1[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data_1,
                    }
                    new_name_list_1.append(name)
                    length_list_1.append(len(motion))
                    count += 1            

        name_list_1, length_list_1 = zip(
            *sorted(zip(new_name_list_1, length_list_1), key=lambda x: x[1]))

        if progress_bar:
            enumerator_2 = enumerate(
                track(
                    id_list_2,
                    f"Loading HumanML3D {split_subfile_2.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator_2 = enumerate(id_list_2)

        count = 0
        bad_count = 0
        new_name_list_2 = []
        length_list_2 = []

        for i, name in enumerator_2:
            if count > maxdata:
                break
            motion = np.load(pjoin(motion_dir, name + ".npy"))
            if (len(motion)) < self.min_motion_length or (len(motion) >=200):
                bad_count += 1
                continue
            text_data_2 = []
            flag = True

            # name = style_to_neutral[name]
            text_path = pjoin(text_dir, name + ".txt")
            assert os.path.exists(text_path)
            with cs.open(text_path) as f:
                for line in f.readlines():
                    text_dict_2 = {}
                    line_split = line.strip().split("#")
                    caption = line_split[0]
                    text_dict_2["caption"] = caption
                    text_data_2.append(text_dict_2)

                if flag:
                    data_dict_2[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data_2,
                    }
                    new_name_list_2.append(name)
                    length_list_2.append(len(motion))
                    count += 1            

        name_list_2, length_list_2 = zip(
            *sorted(zip(new_name_list_2, length_list_2), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr_1 = np.array(length_list_1)
        self.data_dict_1 = data_dict_1
        self.name_list_1 = name_list_1

        self.length_arr_2 = np.array(length_list_2)
        self.data_dict_2 = data_dict_2
        self.name_list_2 = name_list_2

        self.nfeats = motion.shape[1]
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length

        self.pointer_1 = np.searchsorted(self.length_arr_1, length)
        print("Pointer Pointing at %d" % self.pointer_1)

        # self.pointer_2 = np.searchsorted(self.length_arr_2, length)
        # print("Pointer Pointing at %d" % self.pointer_2)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def transform(self,data):
        return (data - self.mean) / self.std

    def get_mean_std(self):
        return self.mean, self.std
    
    def __len__(self):
        return len(self.name_list_1) - self.pointer

    def __getitem__(self, item):
        idx_1 = (self.pointer_1 + item) #% len(self.name_list_1)
        data_1 = self.data_dict_1[self.name_list_1[idx_1]]
        motion_1, m_length_1, text_list_1 = data_1["motion"], data_1["length"], data_1["text"]

        idx_2 = (self.pointer_1 + item + random.randint(0,len(self.name_list_2)-1)) % len(self.name_list_2)
        data_2 = self.data_dict_2[self.name_list_2[idx_2]]
        motion_2, m_length_2, text_list_2 = data_2["motion"], data_2["length"], data_2["text"]
      
        # Randomly select a caption
        text_data_1 = random.choice(text_list_1)
        caption_1 = text_data_1["caption"]

        text_data_2 = random.choice(text_list_2)
        caption_2 = text_data_2["caption"]

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length_1 = (m_length_1 // self.unit_length - 1) * self.unit_length
            m_length_2 = (m_length_2 // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length_1 = (m_length_1 // self.unit_length) * self.unit_length
            m_length_2 = (m_length_2 // self.unit_length) * self.unit_length
        
        m_length_1 = min(196,m_length_1)
        m_length_2 = min(196,m_length_2)
        idx_1 = random.randint(0, len(motion_1) - m_length_1)
        motion_1 = motion_1[idx_1:idx_1 + m_length_1]
        "Z Normalization"
        motion_1 = (motion_1 - self.mean) / self.std


        idx_2 = random.randint(0, len(motion_2) - m_length_2)
        motion_2 = motion_2[idx_2:idx_2 + m_length_2]
        "Z Normalization"
        motion_2 = (motion_2 - self.mean) / self.std

        # debug check nan
        if np.any(np.isnan(motion_1)):
            raise ValueError("nan in motion")

        return (
            caption_1,
            motion_1,
            m_length_1,

            caption_2,
            motion_2,
            m_length_2,
        )


class Text2MotionDataset(data.Dataset):

    def __init__(
        self,
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        text_dir,
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict = {}
        id_list = []
        
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading HumanML3D {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)

        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []

        for i, name in enumerator:
            if count > maxdata:
                break
            motion = np.load(pjoin(motion_dir, name + ".npy"))
            if (len(motion)) < self.min_motion_length or (len(motion) >=200):
                bad_count += 1
                continue
            text_data = []
            flag = True

            text_path = pjoin(text_dir, name + ".txt")
            assert os.path.exists(text_path)
            with cs.open(text_path) as f:
                for line in f.readlines():
                    text_dict = {}
                    line_split = line.strip().split("#")
                    caption = line_split[0]

                    text_dict["caption"] = caption
                    text_data.append(text_dict)
                
                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    count += 1            

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def transform(self,data):
        return (data - self.mean) / self.std

    def get_mean_std(self):
        return self.mean, self.std
    
    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]

        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        # Randomly select a caption

        text_data = random.choice(text_list)
        caption = text_data["caption"]

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)

        motion = motion[idx:idx + m_length]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # debug check nan
        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        return (
            caption,
            motion,
            m_length,
        )


class MotionDatasetV2(data.Dataset):

    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4:4 + (joints_num - 1) * 3] = std[4:4 +
                                                  (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3:4 +
                (joints_num - 1) * 9] = (std[4 + (joints_num - 1) * 3:4 +
                                             (joints_num - 1) * 9] / 1.0)
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9:4 + (joints_num - 1) * 9 +
                joints_num * 3] = (std[4 + (joints_num - 1) * 9:4 +
                                       (joints_num - 1) * 9 + joints_num * 3] /
                                   1.0)
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = (
                std[4 +
                    (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias)

            assert 4 + (joints_num -
                        1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, "mean.npy"), mean)
            np.save(pjoin(opt.meta_dir, "std.npy"), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(
            len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class RawTextDataset(data.Dataset):

    def __init__(self, opt, mean, std, text_file, w_vectorizer):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.nlp = spacy.load("en_core_web_sm")

        with cs.open(text_file) as f:
            for line in f.readlines():
                word_list, pos_list = self.process_text(line.strip())
                tokens = [
                    "%s/%s" % (word_list[i], pos_list[i])
                    for i in range(len(word_list))
                ]
                self.data_dict.append({
                    "caption": line.strip(),
                    "tokens": tokens
                })

        self.w_vectorizer = w_vectorizer
        print("Total number of descriptions {}".format(len(self.data_dict)))

    def process_text(self, sentence):
        sentence = sentence.replace("-", "")
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == "NOUN"
                    or token.pos_ == "VERB") and (word != "left"):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[item]
        caption, tokens = data["caption"], data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len


class TextOnlyDataset(data.Dataset):

    def __init__(self, opt, mean, std, split_file, text_dir, **kwargs):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 120

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {"text": [text_dict]}
                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {"text": text_data}
                    new_name_list.append(name)
            except:
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        text_list = data["text"]

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        return None, None, caption, None, np.array([0
                                                    ]), self.fixed_length, None
        # fixed_length can be set from outside before sampling


# A wrapper class for t2m original dataset for MDM purposes
class HumanML3D(data.Dataset):

    def __init__(self,
                 mode,
                 datapath="./dataset/humanml_opt.txt",
                 split="train",
                 **kwargs):
        self.mode = mode

        self.dataset_name = "t2m"
        self.dataname = "t2m"

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f"."
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = (
            None  # torch.device('cuda:4') # This param is not in use in this context
        )
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        self.opt = opt
        print("Loading dataset %s ..." % opt.dataset_name)

        if mode == "gt":
            # used by T2M models (including evaluators)
            self.mean = np.load(pjoin(opt.meta_dir, "mean.npy"))
            self.std = np.load(pjoin(opt.meta_dir, "std.npy"))
        elif mode in ["train", "eval", "text_only"]:
            # used by our models
            self.mean = np.load(pjoin(opt.data_root, "Mean.npy"))
            self.std = np.load(pjoin(opt.data_root, "Std.npy"))

        if mode == "eval":
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, "mean.npy"))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, "std.npy"))

        self.split_file = pjoin(opt.data_root, f"{split}.txt")
        if mode == "text_only":
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std,
                                               self.split_file)
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, "glove"),
                                               "our_vab")
            self.t2m_dataset = Text2MotionDatasetV2(self.opt, self.mean,
                                                    self.std, self.split_file,
                                                    self.w_vectorizer)
            self.num_actions = 1  # dummy placeholder

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()
