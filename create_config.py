import configparser

config = configparser.ConfigParser()

config[f'cropped_exp'] = {
    'src_dir': "/home/shahafb/Action_Detection_project/small_dataset",
    'crop_range': "0,2399",
    'backbone_size': "base", # in ["small", "base", "large", "giant"]
    'batch_size': 8,
    'lr': 10**(-6),
    'num_epochs': 5,
    'epsilon' : 0.00001
}

with open('argsconfig.ini', 'w') as configfile:
  config.write(configfile)