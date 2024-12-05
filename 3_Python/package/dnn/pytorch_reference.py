import numpy as np

from package.dnn.pytorch_dataclass import Config_Dataset
from package.digital.fex import SettingsFeature, FeatureExtraction
from package.digital.cluster import SettingsCluster, Clustering
from package.plot.plot_dnn import plot_3d_featspace, translate_feats_into_list
from package.plot.plot_metric import plot_confusion


def generate_reference_cluster(config_feat: SettingsFeature, config_clus: SettingsCluster, config_data: Config_Dataset,
                               take_num_samples: int = -1, do_reordering=False, path2save: str = '') -> None:
    """"""
    data_used = config_data.load_dataset()
    data_orig = data_used['data']
    data_dict = data_used['dict']
    true_ids = data_used['label']

    feats = FeatureExtraction(config_feat).pca(data_orig)
    hndl_clus = Clustering(config_clus)
    pred_ids = hndl_clus.init(feats)

    mark_feat0 = translate_feats_into_list(feats, pred_ids, take_num_samples)
    plot_3d_featspace(pred_ids, mark_feat0, [0, 1, 2], data_classname=data_dict)
    mark_feat1 = translate_feats_into_list(feats, true_ids, take_num_samples)
    plot_3d_featspace(true_ids, mark_feat1, [0, 1, 2], data_classname=data_dict, show_plot=False)

    if do_reordering:
        pred_ids_reordered = hndl_clus.sort_pred2label_data(pred_ids, true_ids, feats, take_num_samples)
    else:
        pred_ids_reordered = pred_ids
    plot_confusion(true_ids, pred_ids_reordered, show_accuracy=True,
                   cl_dict=data_dict, path2save=path2save, show_plots=True)


def make_reordered_confusion_matrix(config_data: Config_Dataset, num_pred_ids: list, path2save: str = '') -> None:
    """"""
    data_used = config_data.load_dataset()
    data_dict = data_used['dict']
    true_ids = data_used['label']

    cluster_avai, cluster_samples = np.unique(true_ids, return_counts=True)
    for idx, num_cluster in enumerate(num_pred_ids):
        true = np.zeros((sum(num_cluster),), dtype=np.uint8) + cluster_avai[idx]
        true_ids_new = true if idx == 0 else np.concatenate((true_ids_new, true), axis=0)

        for idy, num_samples in enumerate(num_cluster):
            pred = np.zeros((num_samples,), dtype=np.uint8) + cluster_avai[idy]
            if idx == 0 and idy == 0:
                pred_ids = pred
            else:
                pred_ids = np.concatenate((pred_ids, pred), axis=0)

    plot_confusion(true_ids_new, pred_ids, show_accuracy=True,
                   cl_dict=data_dict, path2save=path2save, show_plots=True)


if __name__ == "__main__":
    sets_feature = SettingsFeature(
        no_features=3
    )
    sets_cluster = SettingsCluster(
        type='kMeans',
        no_cluster=6
    )
    sets_data = Config_Dataset(
        data_path='data',
        data_file_name='quiroga',
        use_cell_library=0,
        augmentation_do=False,
        augmentation_num=0,
        normalization_do=True,
        normalization_method='minmax',
        reduce_samples_per_cluster_do=False,
        reduce_samples_per_cluster_num=0,
        exclude_cluster=[]
    )

    num_fake_ids = [[1000, 0, 0, 0, 0, 0], [0, 1000, 0, 0, 0, 0], [0, 0, 1000, 0, 0, 0],
                    [0, 0, 0, 1300, 10, 7], [0, 0, 0, 6, 1400, 4], [0, 0, 0, 5, 3, 950]]
    make_reordered_confusion_matrix(sets_data, num_fake_ids)

    generate_reference_cluster(sets_feature, sets_cluster, sets_data, 2000)
