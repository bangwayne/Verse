from torch.utils.data import Sampler
import numpy as np
from torch.utils.data import DistributedSampler, Sampler
import numpy as np
import torch.distributed as dist
import math

class LabelBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.label_to_indices = self._map_labels_to_indices()

    def _map_labels_to_indices(self):
        # Map each label to a list of indices that have that label
        label_to_indices = {}
        for idx, sample in enumerate(self.dataset):
            label = sample['target']['labels'].item()  # Assuming label is a tensor with one item
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices

    def __iter__(self):
        # Shuffle each label's indices
        for indices in self.label_to_indices.values():
            np.random.shuffle(indices)

        # Generate batches ensuring each batch is from a single label
        batches = []
        for indices in self.label_to_indices.values():
            batches.extend([
                indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)
            ])

        # Shuffle batches to ensure different order of label batches each epoch
        np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        # Calculate the total number of complete batches
        return sum(len(indices) // self.batch_size for indices in self.label_to_indices.values())



class DistributedLabelBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas if num_replicas is not None else dist.get_world_size()
        self.rank = rank if rank is not None else dist.get_rank()
        self.label_to_indices = self._map_labels_to_indices()
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def _map_labels_to_indices(self):
        # Map each label to a list of indices that have that label
        label_to_indices = {}
        for idx, sample in enumerate(self.dataset):
            label = sample['target']['labels'].item()  # Assuming label is a tensor with one item
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices

    def __iter__(self):
        # Partition each label's indices
        partitioned_by_label = {}
        for label, indices in self.label_to_indices.items():
            np.random.shuffle(indices)
            # Splitting indices by the total number of replicas
            partitioned_by_label[label] = [indices[i::self.num_replicas] for i in range(self.num_replicas)]

        local_batches = []
        # Generate local batches for this specific rank
        for indices in partitioned_by_label.values():
            local_indices = indices[self.rank]  # Select indices for the current rank
            local_batches.extend([
                local_indices[i:i + self.batch_size] for i in range(0, len(local_indices), self.batch_size)
            ])

        # Shuffle local batches to mix labels within the same rank
        np.random.shuffle(local_batches)

        # Yield batches
        for batch in local_batches:
            yield batch

    def __len__(self):
        # Return the number of batches this sampler can provide
        return self.total_size // self.batch_size