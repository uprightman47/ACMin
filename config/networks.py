from .base import Base, Config


class Default(Config):
    dataset = 'icdm_test' # ogbn_arxiv, ogbn_mag, icdm_test
    @property
    def name(self):
        if self._name:
            return self._name
        return [k for k, v in globals().items() if v is self][0]
    @name.setter
    def name(self, v): self._name = v

    data_total_path = 'data'
    cluster_result_path = 'results'
    feature_form = 'representations'
    # ---------------------------------------------------------------------------- #
    # trainning
    # ---------------------------------------------------------------------------- #
    batch_size = 1

    # ---------------------------------------------------------------------------- #
    # Clustering
    # ---------------------------------------------------------------------------- #
    cluster_parts_num = 10
    ACMin_cluster_params = dict(alpha=0.2, beta = 0.35, t=5, tmax=200, ri=False, print_batch=20)
    num_cluster = 200
    # num_cluster_init = 20
    mode = 'cluster' #cluster/eval_metrics
default = Default()