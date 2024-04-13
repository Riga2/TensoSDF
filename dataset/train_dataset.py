from torch.utils.data import Dataset

from dataset.database import get_database_split, parse_database_name
class DummyDataset(Dataset):
    default_cfg={
        'database_name': '',
        'split_manul': False,
    }
    def __init__(self, cfg, is_train, dataset_dir):
        self.cfg={**self.default_cfg,**cfg}
        if not is_train:
            database = parse_database_name(self.cfg['database_name'], dataset_dir)
            split_manul = False
            if self.cfg['split_manul']:
                split_manul = True
            train_ids, test_ids = get_database_split(database, 'validation', split_manul)
            self.train_num = len(train_ids)
            self.test_num = len(test_ids)
        self.is_train = is_train

    def __getitem__(self, index):
        if self.is_train:
            return {}
        else:
            return {'index': index}

    def __len__(self):
        if self.is_train:
            return 99999999
        else:
            return self.test_num