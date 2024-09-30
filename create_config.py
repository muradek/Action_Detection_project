import configparser

config = configparser.ConfigParser()

config[f'checking_save_options'] = {
    'src_dir': "/home/muradek/project/Action_Detection_project/data/small_set_sampled_2024-10-01_00:49:24",
    'sample_frequency': 200,
    'backbone_size': "base", # in ["small", "base", "large", "giant"]
    'batch_size': 8,
    'lr': 10**(-6),
    'num_epochs': 1
}

with open('argsconfig.ini', 'w') as configfile:
  config.write(configfile)