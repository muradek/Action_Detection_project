import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import configparser
from datetime import datetime
import os

from models import *
from datasets import *
from train import *


def main():
    # split data to train and test
    # src_dir = "/home/muradek/project/Action_Detection_project/data/full_dataset"
    # train_dir = "/home/muradek/project/Action_Detection_project/data/train_data"
    # test_dir = "/home/muradek/project/Action_Detection_project/data/test_data"
    # train_size = 50
    # test_size = 15
    # split_data(src_dir, train_dir, test_dir, train_size, test_size)

    # smaple data (train/test)
    # src_dir = "/home/muradek/project/Action_Detection_project/data/train_data"
    # train_crop_range = [750, 1550]
    # transform = transforms.Compose([
    # transforms.Resize((392, 798)),
    # transforms.ToTensor()])
    # FramesDataset(src_dir, crop_range=crop_range, transform=transform)

    # train dino model (also creates FrameDataset and saves it locally)
    dino_model = train_dino_model("argsconfig.ini")

    # create embeddings (1. for taining LSTM model, 2. for testing LSTM model)
    # backbone_size = "base"
    # state_dict_path = "/home/muradek/project/Action_Detection_project/tuned_DINO_models/finetuned_base_24000frames_4epochs_11-24_20:03.pth"
    # src_dir = "/home/muradek/project/Action_Detection_project/data/test_10"
    # dst_dir = "/home/muradek/project/Action_Detection_project/embeddings"
    # crop_range = [0, 2400]
    # embeddings_dir = create_embeddings(backbone_size, state_dict_path, src_dir, dst_dir, crop_range)
    # print("Embeddings created at: ", embeddings_dir)

    # train LSTM model
    # src_dir = "/home/muradek/project/Action_Detection_project/embeddings/11-24_22:27_24000embeddings"
    # lstm_model = train_lstm_model(backbone_size="base", src_dir=src_dir, sequence_length=15, crop_range=[750,1550], epsilon=1e-05)

    # load LSTM model
    # lstm_model = LSTM(embedding_dim=768, hidden_size=256, num_layers=3, sequence_length=15, num_classes=11)
    # fine_tuned_lstm_path = "/home/muradek/project/Action_Detection_project/tuned_LSTM_models/LSTM_base_24000embeddings_4epochs_11-25_12:48.pth"
    # lstm_model.load_state_dict(torch.load(fine_tuned_lstm_path))

    # # test LSTM model
    # lstm_model.eval()
    # embeddings_dir = "/home/muradek/project/Action_Detection_project/embeddings/11-25_14:15_test_10_24000embeddings"
    # dataset = EmbeddingsDataset(embeddings_dir, sequence_length=15, crop_range=[0, 2400])
    # dataloader = DataLoader(dataset, batch_size=24, shuffle=False, num_workers=0)
    # pred_outputs = torch.empty(0).cuda()
    # labels = torch.empty(0).cuda()
    # with torch.no_grad():
    #     for seq_batch, label_batch, _ in dataloader:
    #         seq_batch = seq_batch.cuda()
    #         pred_output = lstm_model(seq_batch)
    #         pred_outputs = torch.cat((pred_outputs, pred_output), 0)

    #         label_batch = label_batch.cuda()
    #         one_hot_labels = torch.argmax(label_batch, dim=1) 
    #         labels = torch.cat((labels, one_hot_labels), 0)

        
    # print("outputs shape is: ", pred_outputs.shape)
    # print("labels shape is: ", labels.shape)
    # curr_time = datetime.now().strftime("%m-%d_%H:%M")
    # output_size = dataset.__len__()
    # output_file = curr_time + "_" + str(output_size) + "lstm_outputs.pth"
    # labels_file = curr_time + "_" + str(output_size) + "labels.pth"
    # outputs_path = f"/home/muradek/project/Action_Detection_project/outputs/{output_file}"
    # labels_path = f"/home/muradek/project/Action_Detection_project/outputs/{labels_file}"
    # torch.save(pred_outputs, outputs_path)
    # torch.save(labels, labels_path)

    # output = torch.load(outputs_path)
    # print("models output:")
    # for i in range(11):
    #     sum = torch.sum(output == i)
    #     print(f"sum of {i} is: {sum}")

    # labels = torch.load(labels_path)
    # print("labels:")
    # for i in range(11):
    #     sum = torch.sum(labels == i)
    #     print(f"sum of {i} is: {sum}")

    # num_gpus = torch.cuda.device_count()
    # print(f"Allocated GPUs: {num_gpus}")
    # gpus = os.getenv("CUDA_VISIBLE_DEVICES", "")
    # num_gpus = len(gpus.split(",")) if gpus else 0
    # print(f"Allocated GPUs: {num_gpus} ({gpus})")

if __name__ == "__main__":
    main()

"""
code snippets
1. Memory management
print(torch.cuda.memory_summary())

param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))


2. pipelines (prepare_data.py)
src_dir = "/home/muradek/project/Action_Detection_project/small_set"
crop_range = [700, 1700]
dataset = FramesDataset(src_dir, crop_range, transform)

3. Freezing DINO
for param in model.backbone_model.parameters():
    param.requires_grad = False


for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

"""