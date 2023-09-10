from .base_network import ogbn_arxivDataset, ogbn_magDataset, icdmDataset
#, ogbn_magDataset, AmazonProductsDataset

def get_dataset(cfg):
    Dataset = {
        'ogbn_arxiv': ogbn_arxivDataset,
        'ogbn_mag': ogbn_magDataset,
        'icdm_test': icdmDataset
    }[cfg.dataset]
    dataset = Dataset(cfg)
    return dataset
