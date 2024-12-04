from package.dnn.pytorch_dataclass import Config_Dataset
from package.digital.fex import SettingsFeature, FeatureExtraction
from package.digital.cluster import SettingsCluster, Clustering
from package.plot.plot_dnn import plot_3d_featspace, translate_feats_into_list
from package.plot.plot_metric import plot_confusion


def generate_reference_cluster(config_feat: SettingsFeature, config_clus: SettingsCluster,
                               config_data: Config_Dataset, take_num_samples: int = -1, path2save: str = '') -> None:
    """"""
    data_used = config_data.load_dataset()
    data_orig = data_used['data']
    data_dict = data_used['dict']
    true_ids = data_used['label']

    feats = FeatureExtraction(config_feat).pca(data_orig)
    hndl_clus = Clustering(config_clus)
    pred_ids = hndl_clus.init(feats)

    mark_feat0 = translate_feats_into_list(feats, pred_ids, take_num_samples)
    plot_3d_featspace(pred_ids, mark_feat0, [0, 1, 2])
    mark_feat1 = translate_feats_into_list(feats, true_ids, take_num_samples)
    plot_3d_featspace(true_ids, mark_feat1, [0, 1, 2], show_plot=True)

    pred_ids_reordered = hndl_clus.sort_pred2label_data(pred_ids, true_ids, feats, take_num_samples)
    plot_confusion(true_ids, pred_ids_reordered, show_accuracy=True,
                   cl_dict=data_dict, path2save=path2save, show_plots=True)


if __name__ == "__main__":
    # --- Settings
    sets_feature = SettingsFeature(
        no_features=5
    )
    sets_cluster = SettingsCluster(
        type='kMeans',
        no_cluster=4
    )
    sets_data = Config_Dataset(
        data_path='data',
        data_file_name='rgc_mcs',
        use_cell_library=0,
        augmentation_do=False,
        augmentation_num=0,
        normalization_do=False,
        normalization_method='minmax',
        reduce_samples_per_cluster_do=False,
        reduce_samples_per_cluster_num=0,
        exclude_cluster=[4]
    )

    generate_reference_cluster(sets_feature, sets_cluster, sets_data, 2000)
