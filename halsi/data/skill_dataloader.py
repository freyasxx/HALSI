from torch.utils.data import Dataset
from halsi.utils.general_utils import AttrDict
import numpy as np
import os


class SkillsDataset(Dataset):
    SPLIT = AttrDict(train=0.99, val=0.01, test=0.0)

    def __init__(self, dataset_name, phase, transform):
        self.phase = phase
        self.transform = transform

        curr_dir = os.path.dirname(__file__)
        fname = os.path.join(curr_dir, "../../dataset/" + dataset_name + "/demos.npy")

        self.seqs = np.load(fname, allow_pickle=True)
        self.n_seqs = len(self.seqs)
        print("Dataset size: ", self.n_seqs)

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        elif self.phase == "test":
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs

    def __getitem__(self, index):
        seq = self.seqs[self.start + index]
        actions = np.array(seq.actions, dtype=np.float32)
        obs = np.array(seq.obs, dtype=np.float32)
        output = AttrDict(obs=obs, actions=actions)
        return output

    def __len__(self):
        return self.end - self.start
        
