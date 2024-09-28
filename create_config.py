import configparser

config = configparser.ConfigParser()

config['DEFAULT'] = {
    'src_dir': "/home/muradek/project/Action_Detection_project/small_set",
    'sample_frequency': 100,
    'backbone_size': "base", # in ["small", "base", "large", "giant"]
    'batch_size': 8,
    'lr': 0.001,
    'num_epochs': 3
}


with open('argsconfig.ini', 'w') as configfile:
  config.write(configfile)