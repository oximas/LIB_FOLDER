from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/content/data/got10k_lmdb'
    settings.got10k_path = '/content/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/content/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/content/data/lasot_lmdb'
    settings.lasot_path = '/content/data/lasot'
    settings.lasotlang_path = '/content/data/lasot'
    settings.network_path = '/content/UETrack/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/content/data/nfs'
    settings.otb_path = '/content/data/OTB2015'
    settings.otblang_path = '/content/data/otb_lang'
    settings.prj_dir = '/content/UETrack'
    settings.result_plot_path = '/content/UETrack/test/result_plots'
    settings.results_path = '/content/UETrack/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/content/UETrack'
    settings.segmentation_path = '/content/UETrack/test/segmentation_results'
    settings.tc128_path = '/content/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/content/data/tnl2k/test'
    settings.tpl_path = ''
    settings.trackingnet_path = '/content/data/trackingnet'
    settings.uav_path = '/content/data/UAV123'
    settings.vot_path = '/content/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

