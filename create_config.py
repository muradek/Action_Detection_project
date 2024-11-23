import configparser

config = configparser.ConfigParser()

config[f'cropped_exp'] = {
    'src_dir': "/home/muradek/project/Action_Detection_project/data/train_data_sampled_2024-11-23_18:06:41",
    'crop_range': "700,1700",
    'backbone_size': "base", # in ["small", "base", "large", "giant"]
    'batch_size': 8,
    'lr': 10**(-6),
    'num_epochs': 5,
    'epsilon' : 0.00001
}

with open('argsconfig.ini', 'w') as configfile:
  config.write(configfile)