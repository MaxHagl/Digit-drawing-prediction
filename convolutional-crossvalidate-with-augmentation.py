from skimage import io
import torch
import pandas as pd
import torch.nn as nn
import du.lib as dulib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torchvision import transforms


# --- Load and build dataset (unchanged) ---
digits = io.imread('/Users/maximilianhagl/Library/CloudStorage/OneDrive-Personal/College/Junior 25_26/intro_to_machine_learing/Project 3/digits.png')

xss = torch.Tensor(5000,400)
idx = 0
for i in range(0, 1000, 20):
  for j in range(0, 2000, 20):
    xss[idx] = torch.Tensor((digits[i:i+20,j:j+20]).flatten())
    idx = idx + 1

xss = xss.float() / 255.0

yss = torch.LongTensor(len(xss))
for i in range(len(yss)):
  yss[i] = i//500

# --- Data augmentation: jitter position & scale ---
augment_transform = transforms.RandomAffine(
    degrees=30,                 # no rotation
    translate=(0.2, 0.2),    # up to ±15% (~±3 px on 20x20)
    scale=(0.85, 1.15),          # zoom between 90% and 110%
    fill=0                     # background fill (0 if your background is black)
)

def augment_dataset(X, y, n_aug=2):
    """
    X: [N, 400] tensor (flattened 20x20)
    y: [N] tensor
    n_aug: how many augmented copies to add per original sample

    returns: X_aug, y_aug with (1 + n_aug)*N samples
    """
    # reshape to [N, 1, 20, 20] for transforms
    X_imgs = X.view(-1, 1, 20, 20)
    X_aug_list = [X_imgs]
    y_aug_list = [y]

    for _ in range(n_aug):
        aug_imgs = []
        for img in X_imgs:
            # img: [1, 20, 20], already float and normalized (0–1)
            aug_img = augment_transform(img)
            aug_imgs.append(aug_img)
        aug_imgs = torch.stack(aug_imgs, dim=0)   # [N, 1, 20, 20]
        X_aug_list.append(aug_imgs)
        y_aug_list.append(y.clone())

    X_all = torch.cat(X_aug_list, dim=0).view(-1, 400)  # back to [N*, 400]
    Y_all = torch.cat(y_aug_list, dim=0)
    return X_all, Y_all

torch.manual_seed(42)
n_classes = 10
n_per_class = 500
n_train = int(0.8 * n_per_class)   # 400
n_test  = n_per_class - n_train     # 100

perms = [torch.randperm(n_per_class) for _ in range(n_classes)]

train_idx = torch.cat([perms[c][:n_train] + c * n_per_class for c in range(n_classes)])
test_idx  = torch.cat([perms[c][n_train:] + c * n_per_class for c in range(n_classes)])

# shuffle within splits
train_idx = train_idx[torch.randperm(train_idx.numel())]
test_idx  = test_idx[torch.randperm(test_idx.numel())]

xss_train = xss[train_idx]
yss_train = yss[train_idx]
xss_test  = xss[test_idx]
yss_test  = yss[test_idx]

print(f"Train size: {len(xss_train)} (400/class x 10 = 4000)")
print(f"Test  size: {len(xss_test)}  (100/class x 10 = 1000)")
# --- Models (unchanged) ---
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.layer1 = nn.Linear(400, 1)
    def forward(self, x):
        return self.layer1(x)

class nonLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(400, 200)
        self.layer2 = nn.Linear(200, 100)
        self.layer3 = nn.Linear(100, 50)
        self.layer4 = nn.Linear(50, 25)
        self.layer5 = nn.Linear(25, 12)
        self.layer6 = nn.Linear(12, 6)
        self.output = nn.Linear(6, 1)
    def forward(self, xss):
        x = torch.relu(self.layer1(xss))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = torch.relu(self.layer6(x))
        return self.output(x)

class SigmoidModel(nn.Module):

  def __init__(self):
    super(SigmoidModel, self).__init__()
    self.layer1 = nn.Linear(400,200)
    self.layer2= nn.Linear(200,100)
    self.layer3= nn.Linear(100,1)

  def forward(self, x):
    x = torch.relu(self.layer1(x))
    x = torch.relu(self.layer2(x))
    x = self.layer3(x)
    return torch.sigmoid(x)
  

class LogSoftmaxModel(nn.Module):

    def __init__(self):
        super(LogSoftmaxModel, self).__init__()
        self.layer1 = nn.Linear(400,200)
        self.layer2 = nn.Linear(200,100)
        self.layer3 = nn.Linear(100,10)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
     

        return torch.log_softmax(x, dim=1)
    
class ConvolutionalModel(nn.Module):

  def __init__(self):
    super(ConvolutionalModel, self).__init__()
    self.meta_layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding = 2),
        nn.BatchNorm2d(16),  #stabilizes neural net 
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
        nn.Dropout(0.05),
         # protects a bit agianst over fitting 

        nn.Conv2d(16, 32, 3, 1, 1),  # NEW block
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.10),

        nn.Conv2d(32, 64, 3, 1, 1),  # NEW block
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.15),
        

        nn.Conv2d(64, 128, 3, 1, 1),  # NEW block
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.20),

        
    )

    
    self.fc_layer1 = nn.Linear(512,64)
    self.dropout_fc = nn.Dropout(0.1)

    self.fc_layer2 = nn.Linear(64, 10)

  def forward(self, xss):
    xss = xss.view(-1, 1, 20, 20)
    xss = self.meta_layer1(xss)
    xss = torch.reshape(xss, (-1, 512))
    xss = torch.relu(self.fc_layer1(xss))
    xss = self.fc_layer2(xss)
    return (xss)



model = ConvolutionalModel()
batch_size = 10
learning_rate = 0.001
momentum = 0.9
epochs = 75
wd=0.0001



def eval_accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        return (preds == y).float().mean().item() * 100.0

K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

fold_acc = []

for fold, (train_idx, val_idx) in enumerate(skf.split(xss, yss), start=1):
    X_tr, y_tr = xss[train_idx], yss[train_idx]
    X_va, y_va = xss[val_idx], yss[val_idx]

    # --- Augment TRAIN data only ---
    X_tr, y_tr = augment_dataset(X_tr, y_tr, n_aug=2)
    print(f"[Fold {fold}] After augmentation: {len(X_tr)} train samples")

    model = ConvolutionalModel()
    model = dulib.train(
        model,
        crit=nn.CrossEntropyLoss(),
        gpu=(-1,),
        train_data=(X_tr, y_tr),
        valid_data=(X_va, y_va),
        valid_metric=True,
        learn_params={'lr': learning_rate, 'mo': momentum},
        bs=batch_size,
        epochs=epochs,
        graph=0
    )

    acc = eval_accuracy(model, X_va, y_va)
    fold_acc.append(acc)
    print(f"[Fold {fold}/{K}] val_acc = {acc:.2f}%")

print(f"\nCV mean acc = {np.mean(fold_acc):.2f}%  (±{np.std(fold_acc):.2f})")

print("saving model ")
torch.save(model.state_dict(), 'crossvalidate_9.pyt')
print("done")


print("total examples:", len(xss_train), "; batch size:", batch_size, "; lr:", learning_rate, "; momentum:", momentum)

# --- Evaluate on TEST split only ---
# zero = 0.0
# eight = 1.0
# th = 1e-3
# cutoff = 0.5

# correct = 0
# mis_idx_global = None  # store absolute index in X_test for first misclassification


count = 0



for i in range(len(xss_test)):
    if model(xss_test[i].unsqueeze(0)).argmax(dim=1).item() == yss_test[i].item():
        count += 1
print("Percentage correct on test set:",100*count/len(xss_test))

# for i in range(len(xss)):
#     yhat_val = model(xss[i]).item()
#     y_val = yss[i].item()
#     # same correctness logic you had (tolerant to tiny label noise)
#     is_correct = ((yhat_val > cutoff and abs(y_val - eight) < th) or
#                   (yhat_val < cutoff and abs(y_val - zero)  < th))
#     if is_correct:
#         correct += 1
#     else:
#         if mis_idx_global is None:
#             mis_idx_global = i

# print("Test accuracy (% correct):", 100 * correct / len(xss))

# # --- Show first misclassified TEST example (20x20) ---
# if mis_idx_global is not None:
#     print(f"\nFirst misclassified test sample at test index {mis_idx_global}:")
#     true_label = int(yss_test[mis_idx_global].item())
#     pred_val = model(xss_test[mis_idx_global]).item()
#     print(f" True label: {true_label}")
#     print(f" Predicted value: {pred_val:.4f} (cutoff={cutoff:.2f})")

#     # reshape to 20x20 and print as tensor
#     img_tensor = xss_test[mis_idx_global].reshape(20, 20)
#     print("\n20x20 grayscale tensor of misclassified digit (from TEST set):\n")
#     print(img_tensor)
# else:
#     print("\nNo misclassifications on the test set!")
# --- Confusion matrix (counts + normalized) ---
