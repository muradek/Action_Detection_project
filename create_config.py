import configparser

config = configparser.ConfigParser()

config[f'cropped_exp'] = {
    'src_dir': "/home/muradek/project/Action_Detection_project/data/train_30_sampled_2024-11-24_14:35:11",
    'crop_range': "750,1550",
    'backbone_size': "base", # in ["small", "base", "large", "giant"]
    'batch_size': 8,
    'lr': 10**(-5),
    'num_epochs': 4,
    'epsilon' : 0.00001
}

with open('argsconfig.ini', 'w') as configfile:
  config.write(configfile)