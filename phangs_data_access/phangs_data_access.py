from phangs_data_access.phot_access import PhotAccess
from phangs_data_access.sample_access import SampleAccess
from phangs_data_access.cluster_cat_access import ClusterCatAccess


class PhangsDataAccess(PhotAccess, SampleAccess, ClusterCatAccess):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
