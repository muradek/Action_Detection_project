import torch
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix
import matplotlib.pyplot as plt

gold_labels_path = "/home/muradek/project/Action_Detection_project/outputs/11-25_16:41_24000labels.pth"
pred_labels_path = "/home/muradek/project/Action_Detection_project/outputs/11-25_16:41_24000lstm_outputs.pth"

gold_labels = torch.load(gold_labels_path).cpu().numpy()
pred_labels = torch.load(pred_labels_path).cpu().numpy()

macro_f1 = f1_score(gold_labels, pred_labels, average='macro')
micro_f1 = f1_score(gold_labels, pred_labels, average='micro')
confusion_matrix = confusion_matrix(gold_labels, pred_labels)

disp_label = ['background', 'Perch', 'Lift', 'Reach', 'Grab_nonPellet',
       'Grab', 'Sup', 'AtMouth', 'AtMouth_nonPellet', 'BackPerch', 'Table']
disp = ConfusionMatrixDisplay.from_predictions(gold_labels, pred_labels, display_labels=disp_label, cmap='Greens', normalize=None)
for text in disp.ax_.texts:
    text.set_fontsize(8)
plt.title('Confusion Matrix')
plt.xticks(rotation=90, fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig("/home/muradek/project/Action_Detection_project/outputs/confusion_matrix_3.png")
plt.clf()

print(f"{macro_f1=}")
print(f"{micro_f1=}")
print(f"{confusion_matrix=}")