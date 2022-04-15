import torch
from collections import Counter

def get_dataset(train_data, train_label, change = False):
  if change:
    return My_RE_Dataset(train_data, train_label)
  else:
    return RE_Dataset(train_data, train_label)

class My_RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""

  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    ids = item['input_ids']

    index = (ids >= 32000).nonzero(as_tuple=True)
    # print(index)
    """
    item['index'] = index

    cri = index[0][0]
    other = index[0][2]

    if ids[cri] <= 32002:
      item['SUB'] = int(cri)
      item['OBJ'] = int(other)
    else:
      item['OBJ'] = int(cri)
      item['SUB'] = int(other)
    """

    item['SUB'] = int((ids==32000).nonzero(as_tuple=False))
    item['OBJ'] = int((ids==32002).nonzero(as_tuple=False))

    return item

  def __len__(self):
    return len(self.labels)

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels
    self.label_counter = self._get_label_counter()

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

  def get_n_per_labels(self):
      return [self.label_counter[i] for i in range(30)]

  def _get_label_counter(self):
      label_counter = Counter(self.labels)
      return label_counter