import torch
import torch.optim as optim
import argparse
from typing import List
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import pdb
import wandb
from tqdm import tqdm
import os
import time
import yaml
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from halsi.data.skill_dataloader import SkillsDataset

from halsi.utils.general_utils import AttrDict
from halsi.models.diff import dynamicsVAE         
from torch.nn.utils.rnn import pad_sequence
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dynamic_padding_collate(batch):
    """
    Collate function for variable-length sequences, supports dynamic padding and time series sorting
    """
    obs_list = []
    actions_list = []
    seq_lens = []
    for data in batch:
        obs_list.append(torch.tensor(data['obs']))
        actions_list.append(torch.tensor(data['actions']))
        seq_lens.append(len(data['obs']))
    output = AttrDict(
        obs=obs_list,
        actions=actions_list,
        seq_lens=seq_lens
    )
    return output

class ModelTrainer():
    def __init__(self, dataset_name, config_file):
        self.dataset_name = dataset_name
        self.save_dir = "./results/saved_dskill_models/" + dataset_name + "/"
        os.makedirs(self.save_dir, exist_ok=True)
        self.vae_save_path = self.save_dir + "skill_vae.pth"
        
        config_path = "./configs/skill_mdl/block/config.yaml"

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: ", self.device)

        with open(config_path, 'r') as file:
            conf = yaml.safe_load(file)
            conf = AttrDict(conf)
        for key in conf:
            conf[key] = AttrDict(conf[key])        

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]) 
                    
        train_data = SkillsDataset(dataset_name, phase="train", transform=transform)
        val_data   = SkillsDataset(dataset_name, phase="val", transform=transform)

        self.train_loader = DataLoader(
            train_data,
            batch_size = 256,
            shuffle = True,
            drop_last=True,
            prefetch_factor=8,
            num_workers=48,
            pin_memory=True,
            collate_fn=dynamic_padding_collate
        )

        self.val_loader = DataLoader(
            val_data,
            batch_size = 128,
            shuffle = False,
            drop_last=True,
            prefetch_factor=2,
            num_workers=48,
            pin_memory=True,
            collate_fn=dynamic_padding_collate
        )

        self.skill_vae = dynamicsVAE(n_actions=conf.skill_vae.n_actions, 
                                     n_obs=conf.skill_vae.n_obs, 
                                     n_hidden=conf.skill_vae.n_hidden,
                                     n_z=conf.skill_vae.n_z, 
                                     device=self.device).to(self.device)
        
        self.optimizer = optim.Adam(self.skill_vae.parameters(), lr=conf.skill_vae.lr, weight_decay=1e-5)
       
        self.n_epochs = conf.skill_vae.epochs

    def fit(self, epoch):
        self.skill_vae.train()
        running_loss = 0.0
        for i, data in enumerate(self.train_loader):
            obs_list = [obs.to(self.device) for obs in data["obs"]]
            actions_list = [actions.to(self.device) for actions in data["actions"]]
            seq_lens = data["seq_lens"]

            data = AttrDict(obs=obs_list, actions=actions_list)

            # Call init_hidden to keep interface consistent (no operation in Transformer)
            self.skill_vae.init_hidden(len(obs_list))
            self.optimizer.zero_grad()
            output = self.skill_vae(data, seq_lens)
            losses = self.skill_vae.loss(data, output, seq_lens)
            loss = losses.total_loss
            running_loss += loss.item()

            loss.backward()
            # Add gradient clipping, limit gradient norm to 1.0
            grad_norm = torch.nn.utils.clip_grad_norm_(self.skill_vae.parameters(), max_norm=1.0)
            # Record gradient norm after clipping
            vae_grad_norm = torch.norm(torch.stack([param.grad.norm() for param in self.skill_vae.parameters() if param.grad is not None]), 2).item()
            self.optimizer.step()

            if i % 100 == 0:
                wandb.log({'BC Loss_VAE': losses.bc_loss.item()}, epoch)
                wandb.log({'KL_Loss_VAE': losses.kld_loss.item()}, epoch)
                wandb.log({'VAE_grad_norm': vae_grad_norm}, epoch)
            
        train_loss = running_loss / len(self.train_loader.dataset)
        return train_loss

    def validate(self):
        self.skill_vae.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                obs_list = [obs.to(self.device) for obs in data["obs"]]
                actions_list = [actions.to(self.device) for actions in data["actions"]]
                seq_lens = data["seq_lens"]

                data = AttrDict(obs=obs_list, actions=actions_list)

                self.skill_vae.init_hidden(len(obs_list))
                output = self.skill_vae(data, seq_lens)
                losses = self.skill_vae.loss(data, output, seq_lens)

                loss = losses.bc_loss.item()
                running_loss += loss

        val_loss = running_loss / len(self.val_loader.dataset)

        return val_loss

    def train(self):
        print("Training...")
        val_epoch_loss = None
        for epoch in tqdm(range(self.n_epochs)):
            train_epoch_loss = self.fit(epoch)
            if epoch % 5 == 0:
                val_epoch_loss = self.validate()


            wandb.log({'train_loss': train_epoch_loss}, epoch)
            wandb.log({'val_loss': val_epoch_loss}, epoch)

            if epoch % 50 == 0:
                torch.save(self.skill_vae, self.vae_save_path)
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="block/config.yaml")
    parser.add_argument('--dataset_name', type=str, default="fetch_block_40000")        
    args = parser.parse_args()

    seed = 0
    set_seed(seed)
    wandb.init(
        project="trans_skill_mdl",
    )
    wandb.run.name = "tskill_mdl_" + time.asctime()
    wandb.run.save()

    trainer = ModelTrainer(args.dataset_name, args.config_file)
    trainer.train()
