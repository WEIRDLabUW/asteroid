"""
Dataset classes for training.

Classes:
- Dataset: For DPT-style data with context and query
- ImageDataset: For image-based data
- SequenceDataset: For sequence/trajectory data (DAgger-style)
"""

import pickle
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def convert_to_tensor(data, store_gpu=False):
    """Convert numpy array to torch tensor."""
    tensor = torch.from_numpy(np.array(data)).float()
    if store_gpu and torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class for DPT-style data.
    
    Each item contains context (states, actions, next_states, rewards)
    and a query state with optimal action.
    """

    def __init__(self, path, config, data_ratio=1.0):
        self.shuffle = config['shuffle']
        self.horizon = config['horizon']
        self.store_gpu = config['store_gpu']
        self.config = config

        # Handle path as list or single path
        if not isinstance(path, list):
            path = [path]

        self.trajs = []
        for p in path:
            with open(p, 'rb') as f:
                self.trajs += pickle.load(f)
            
        context_states = []
        context_actions = []
        context_next_states = []
        context_rewards = []
        query_states = []
        optimal_actions = []

        for traj in self.trajs:
            context_states.append(traj['context_states'])
            context_actions.append(traj['context_actions'])
            context_next_states.append(traj['context_next_states'])
            context_rewards.append(traj['context_rewards'])
            query_states.append(traj['query_state'])
            optimal_actions.append(traj['optimal_action'])

        context_states = np.array(context_states)
        context_actions = np.array(context_actions)
        context_next_states = np.array(context_next_states)
        context_rewards = np.array(context_rewards)
        if len(context_rewards.shape) < 3:
            context_rewards = context_rewards[:, :, None]
        query_states = np.array(query_states)
        optimal_actions = np.array(optimal_actions)

        self.dataset = {
            'query_states': convert_to_tensor(query_states, store_gpu=self.store_gpu),
            'optimal_actions': convert_to_tensor(optimal_actions, store_gpu=self.store_gpu),
            'context_states': convert_to_tensor(context_states, store_gpu=self.store_gpu),
            'context_actions': convert_to_tensor(context_actions, store_gpu=self.store_gpu),
            'context_next_states': convert_to_tensor(context_next_states, store_gpu=self.store_gpu),
            'context_rewards': convert_to_tensor(context_rewards, store_gpu=self.store_gpu),
        }

        self.zeros = np.zeros(
            config['state_dim'] ** 2 + config['action_dim'] + 1
        )
        self.zeros = convert_to_tensor(self.zeros, store_gpu=self.store_gpu)

    def __len__(self):
        return len(self.dataset['query_states'])

    def __getitem__(self, index):
        res = {
            'context_states': self.dataset['context_states'][index],
            'context_actions': self.dataset['context_actions'][index],
            'context_next_states': self.dataset['context_next_states'][index],
            'context_rewards': self.dataset['context_rewards'][index],
            'query_states': self.dataset['query_states'][index],
            'optimal_actions': self.dataset['optimal_actions'][index],
            'zeros': self.zeros,
        }

        if self.shuffle:
            perm = torch.randperm(self.horizon)
            res['context_states'] = res['context_states'][perm]
            res['context_actions'] = res['context_actions'][perm]
            res['context_next_states'] = res['context_next_states'][perm]
            res['context_rewards'] = res['context_rewards'][perm]

        return res


class ImageDataset(Dataset):
    """Dataset class for image-based data."""

    def __init__(self, paths, config, transform):
        config['store_gpu'] = False
        super().__init__(paths, config)
        self.transform = transform
        self.config = config

        context_filepaths = []
        query_images = []

        for traj in self.trajs:
            context_filepaths.append(traj['context_images'])
            query_image = self.transform(traj['query_image']).float()
            query_images.append(query_image)

        self.dataset.update({
            'context_filepaths': context_filepaths,
            'query_images': torch.stack(query_images),
        })

    def __getitem__(self, index):
        filepath = self.dataset['context_filepaths'][index]
        context_images = np.load(filepath)
        context_images = [self.transform(images) for images in context_images]
        context_images = torch.stack(context_images).float()

        query_images = self.dataset['query_images'][index]

        res = {
            'context_images': context_images,
            'context_states': self.dataset['context_states'][index],
            'context_actions': self.dataset['context_actions'][index],
            'context_next_states': self.dataset['context_next_states'][index],
            'context_rewards': self.dataset['context_rewards'][index],
            'query_images': query_images,
            'query_states': self.dataset['query_states'][index],
            'optimal_actions': self.dataset['optimal_actions'][index],
            'zeros': self.zeros,
        }

        if self.shuffle:
            perm = torch.randperm(self.horizon)
            res['context_images'] = res['context_images'][perm]
            res['context_states'] = res['context_states'][perm]
            res['context_actions'] = res['context_actions'][perm]
            res['context_next_states'] = res['context_next_states'][perm]
            res['context_rewards'] = res['context_rewards'][perm]

        return res


class SequenceDataset(torch.utils.data.Dataset):
    """
    Dataset class for sequence/trajectory data (DAgger-style).
    
    Each item contains a full trajectory with states, actions, 
    expert_actions, rewards, and dones.
    """

    def __init__(self, trajs, config):
        self.shuffle = config['shuffle']
        self.horizon = config['horizon']
        self.store_gpu = config.get('store_gpu', False)
        self.config = config
        self.trajs = trajs
    
    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, index):
        traj = self.trajs[index]
        res = {
            'states': convert_to_tensor(traj['states'], store_gpu=self.store_gpu),
            'actions': convert_to_tensor(traj['actions'], store_gpu=self.store_gpu),
            'expert_actions': convert_to_tensor(traj['expert_actions'], store_gpu=self.store_gpu),
            'rewards': convert_to_tensor(traj['rewards'], store_gpu=self.store_gpu),
            'dones': convert_to_tensor(traj['dones'], store_gpu=self.store_gpu),
        }
        
        # Include optional fields if present
        if 'values' in traj:
            res['values'] = convert_to_tensor(traj['values'], store_gpu=self.store_gpu)
        if 'expert_values' in traj:
            res['expert_values'] = convert_to_tensor(traj['expert_values'], store_gpu=self.store_gpu)
        if 'query_actions' in traj:
            res['query_actions'] = convert_to_tensor(traj['query_actions'], store_gpu=self.store_gpu)
        if 'query_values' in traj:
            res['query_values'] = convert_to_tensor(traj['query_values'], store_gpu=self.store_gpu)

        return res


def collate_fn(batch):
    """
    Collate function for DataLoader that handles variable length sequences.
    
    Pads sequences to the maximum length in the batch and creates attention masks.
    """
    from torch.nn.utils.rnn import pad_sequence
    
    padded_batch = {}
    for key in batch[0]:
        padded_batch[key] = pad_sequence(
            [item[key] for item in batch], 
            batch_first=True
        )
    
    # Create attention mask (1 for valid positions, 0 for padding)
    lengths = torch.tensor([item['states'].shape[0] for item in batch])
    max_len = lengths.max()
    attention_mask = torch.zeros((len(lengths), max_len), dtype=torch.bool)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1
    
    padded_batch['attention_mask'] = attention_mask
    return padded_batch
