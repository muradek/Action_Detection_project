import configparser

config = configparser.ConfigParser()

config[f'hundred_vids_training'] = {
    'src_dir': "/home/muradek/project/Action_Detection_project/data/dataset_100vids_sampled_2024-11-06_14:25:33",
    'crop_point': 700,
    'backbone_size': "base", # in ["small", "base", "large", "giant"]
    'batch_size': 8,
    'lr': 10**(-6),
    'num_epochs': 2,
    'epsilon' : 0.00001
}

with open('argsconfig.ini', 'w') as configfile:
  config.write(configfile)