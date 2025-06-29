#############################################################################################################################################
#Chapter 1: presetups
#############################################################################################################################################

# section: importing libraries
#--------------------------------------------------------------------------------------------------------------------
print("Execution status: Initating program")
import os
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter
print("Execution status: all libraries imported")


# section: seeding for reproducablity
#--------------------------------------------------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()


#############################################################################################################################################
#Chapter 2: defintions
#############################################################################################################################################

# section: saving visualization function
#--------------------------------------------------------------------------------------------------------------------
def save_visualization(image_tensor, pred_mask, true_mask, epoch, index):
    image = image_tensor.cpu().permute(1, 2, 0).numpy()
    pred = torch.sigmoid(pred_mask).cpu().numpy()
    true = true_mask.cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    image = np.clip(image, 0, 1)
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[1].imshow(pred[0], cmap='Reds')
    axs[1].set_title("Predicted Mask")
    axs[2].imshow(true[0], cmap='Greens')
    axs[2].set_title("Ground Truth Mask")
    for ax in axs: ax.axis('off')

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/epoch_{epoch}_sample_{index}.png")
    plt.close()


# section: Router
#--------------------------------------------------------------------------------------------------------------------
class Router(nn.Module):
    def __init__(self, in_channels, num_experts=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)  # Returns weights for each expert


# section: Dataset loader
#--------------------------------------------------------------------------------------------------------------------
class IDRiDDataset(Dataset):
    def __init__(self, image_paths, labels, mask_dirs, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.mask_dirs = mask_dirs
        self.mask_suffixes = ['_MA', '_HE', '_EX', '_SE', '_OD']
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        base_name = os.path.splitext(os.path.basename(self.image_paths[idx]))[0]

        masks = []
        for mask_dir, suffix in zip(self.mask_dirs, self.mask_suffixes):
            mask_path = os.path.join(mask_dir, base_name + suffix + ".tif")
            mask = Image.open(mask_path).convert("L")
            masks.append(np.array(mask))
        mask_stack = np.stack(masks, axis=0)  # Shape: (5, H, W)

        if self.transform:
            augmented = self.transform(image=np.array(image), mask=mask_stack.transpose(1, 2, 0))
            image = augmented['image']
            mask_stack = augmented['mask'].permute(2, 0, 1)  # Back to (5, H, W)
        return image, torch.tensor(label, dtype=torch.long), mask_stack.float()
print("Execution status: transformations")



# section: data augumentation
#--------------------------------------------------------------------------------------------------------------------
transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15, p=0.3),
    A.Affine(translate_percent={"x": 0.05, "y": 0.05},scale=(0.95, 1.05),rotate=(-15, 15),p=0.3),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])


# section: Architecture definition - shared backbone
#--------------------------------------------------------------------------------------------------------------------
class SharedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
    def forward(self, x):
        return self.backbone(x)  # Output shape: [B, 512, H/32, W/32]



# section: Architecture definition - classification
#--------------------------------------------------------------------------------------------------------------------
class ClassificationExpert(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(x)


# section: Architecture definition - segmentation expert
#--------------------------------------------------------------------------------------------------------------------
class SegmentationExpert(nn.Module):
    def __init__(self, in_channels=512, out_channels=5):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),  # 8 → 16
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),          # 16 → 32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),           # 32 → 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),            # 64 → 128
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),  # 128 → 256
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(x)


# section: Multi task model definition
#--------------------------------------------------------------------------------------------------------------------
class MultiTaskModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.shared = SharedBackbone()
        self.router = Router(in_channels=512, num_experts=2)
        self.classifier = ClassificationExpert(512, num_classes)
        self.segmenter = SegmentationExpert()

    def forward(self, x):
        features = self.shared(x)  # Shared feature extraction
        routing_weights = self.router(features)  # Shape: [B, 2]
        class_out = self.classifier(features)  # Shape: [B, num_classes]
        seg_out = self.segmenter(features)     # Shape: [B, 5, H, W]
        class_gate = routing_weights[:, 0].unsqueeze(1)  # Shape: [B, 1]
        seg_gate = routing_weights[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Shape: [B, 1, 1, 1]
        gated_class_out = class_gate * class_out
        gated_seg_out = seg_gate * seg_out
        return gated_class_out, gated_seg_out




# section: Loss function
#--------------------------------------------------------------------------------------------------------------------
def multitask_loss(class_pred, class_target, seg_pred, seg_target, alpha=0.5):
    classification_loss = nn.CrossEntropyLoss()(class_pred, class_target)
    segmentation_loss = nn.BCEWithLogitsLoss()(seg_pred, seg_target)
    return alpha * classification_loss + (1 - alpha) * segmentation_loss

print("Execution status: defining training and evaluation")


# section: traning
#--------------------------------------------------------------------------------------------------------------------
def train(model, dataloader, optimizer,  epoch, device):
    model.train()
    with open("training_log.csv", "a") as f:
        if epoch == 0:  # Assuming epoch starts from 0
            f.write("epoch,batch,loss\n")
    for batch_idx, (images, labels, masks) in enumerate(dataloader):
        images, labels, masks = images.to(device), labels.to(device), masks.to(device)
        optimizer.zero_grad()
        class_out, seg_out = model(images)
        loss = multitask_loss(class_out, labels, seg_out, masks)
        loss.backward()
        optimizer.step()
        with open("training_log.csv", "a") as f:
            f.write(f"{epoch},{batch_idx},{loss.item():.4f}\n")
        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch_idx)




# section: Evaluvation
#--------------------------------------------------------------------------------------------------------------------
def evaluate(model, dataloader, device, epoch):
    model.eval()
    total_correct = 0
    total_samples = 0
    dice_scores = [[] for _ in range(5)]  # For each lesion type
    class_correct = [0] * 5
    class_total = [0] * 5
    with torch.no_grad():
        for batch_idx, (images, labels, masks) in enumerate(dataloader):
            images, labels, masks = images.to(device), labels.to(device), masks.to(device)
            class_out, seg_out = model(images)
            # Classification accuracy per class
            preds = torch.argmax(class_out, dim=1)
            for i in range(len(labels)):
                class_total[labels[i].item()] += 1
                if preds[i].item() == labels[i].item():
                    class_correct[labels[i].item()] += 1
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            # Dice score per lesion
            seg_out = torch.sigmoid(seg_out)
            for i in range(5):  # 5 lesion types
                intersection = (seg_out[:, i] * masks[:, i]).sum(dim=(1, 2))
                union = seg_out[:, i].sum(dim=(1, 2)) + masks[:, i].sum(dim=(1, 2))
                dice = (2. * intersection) / (union + 1e-8)
                dice_scores[i].extend(dice.cpu().numpy())
            # Save visualizations for first batch
            if batch_idx == 0:
                for i in range(min(2, images.size(0))):
                    save_visualization(images[i], seg_out[i], masks[i], epoch, i)
                
                writer.add_images('Images/original', images[:2], epoch)
                for i in range(5):
                    writer.add_images(f'Masks/predicted_type_{i}', seg_out[:2, i:i+1], epoch)
                    writer.add_images(f'Masks/ground_truth_type_{i}', masks[:2, i:i+1], epoch)
    accuracy = total_correct / total_samples
    avg_dice = [sum(scores) / len(scores) for scores in dice_scores]
    class_acc = [correct / total if total > 0 else 0.0 for correct, total in zip(class_correct, class_total)]
    with open("task_metrics_log.csv", "a") as f:
        f.write(f"Epoch {epoch}, Overall Accuracy: {accuracy:.4f}, " +
                ", ".join([f"Class{i}_Acc: {acc:.4f}" for i, acc in enumerate(class_acc)]) + ", " +
                ", ".join([f"Lesion{i}_Dice: {dice:.4f}" for i, dice in enumerate(avg_dice)]) + "\n")
    writer.add_scalar('Accuracy/Overall', accuracy, epoch)
    writer.add_scalar('Dice/Overall', sum(avg_dice) / len(avg_dice), epoch)
    return accuracy, sum(avg_dice) / len(avg_dice)



# section: Save model
#--------------------------------------------------------------------------------------------------------------------
def save_model(model, path="saved_models/multitask_model.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"\t\t\t\tModel saved to {path}")



#############################################################################################################################################
#Chapter 3 Implementation
#############################################################################################################################################

# section: setting path of dataset and loading data
#--------------------------------------------------------------------------------------------------------------------
image_dir = r"D:\Project_Space\5.retina\retina_data\B. Disease Grading\1. Original Images\a. Training Set"
label_csv = r"D:\Project_Space\5.retina\retina_data\B. Disease Grading\2. Groundtruths\a. IDRiD_Disease Grading_Training Labels.csv"
mask_dirs = [
    r"D:\Project_Space\5.retina\retina_data\A. Segmentation\2. All Segmentation Groundtruths\a. Training Set\1. Microaneurysms",
    r"D:\Project_Space\5.retina\retina_data\A. Segmentation\2. All Segmentation Groundtruths\a. Training Set\2. Haemorrhages",
    r"D:\Project_Space\5.retina\retina_data\A. Segmentation\2. All Segmentation Groundtruths\a. Training Set\3. Hard Exudates",
    r"D:\Project_Space\5.retina\retina_data\A. Segmentation\2. All Segmentation Groundtruths\a. Training Set\4. Soft Exudates",
    r"D:\Project_Space\5.retina\retina_data\A. Segmentation\2. All Segmentation Groundtruths\a. Training Set\5. Optic Disc"
]
image_paths = sorted(glob.glob(image_dir + r"\*.jpg"))
label_df = pd.read_csv(label_csv)
filename_to_label = dict(zip(label_df['Image name'], label_df['Retinopathy grade']))
labels = [filename_to_label[os.path.splitext(os.path.basename(p))[0]] for p in image_paths]
print("Execution status: Dataset paths and labels loaded")

val_image_dir = r"D:\Project_Space\5.retina\retina_data\B. Disease Grading\1. Original Images\b. Testing Set"
val_label_csv = r"D:\Project_Space\5.retina\retina_data\B. Disease Grading\2. Groundtruths\b. IDRiD_Disease Grading_Testing Labels.csv"
val_mask_dirs = [
    r"D:\\Project_Space\\5.retina\\retina_data\\A. Segmentation\\2. All Segmentation Groundtruths\\b. Testing Set\\1. Microaneurysms",
    r"D:\\Project_Space\\5.retina\\retina_data\\A. Segmentation\\2. All Segmentation Groundtruths\\b. Testing Set\\2. Haemorrhages",
    r"D:\\Project_Space\\5.retina\\retina_data\\A. Segmentation\\2. All Segmentation Groundtruths\\b. Testing Set\\3. Hard Exudates",
    r"D:\\Project_Space\\5.retina\\retina_data\\A. Segmentation\\2. All Segmentation Groundtruths\\b. Testing Set\\4. Soft Exudates",
    r"D:\\Project_Space\\5.retina\\retina_data\\A. Segmentation\\2. All Segmentation Groundtruths\\b. Testing Set\\5. Optic Disc"
]
val_image_paths = sorted(glob.glob(val_image_dir + r"\\*.jpg"))
val_label_df = pd.read_csv(val_label_csv)
val_filename_to_label = dict(zip(val_label_df['Image name'], val_label_df['Retinopathy grade']))
val_labels = [val_filename_to_label[os.path.splitext(os.path.basename(p))[0]] for p in val_image_paths]


# section: Choosing GPU
#--------------------------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Execution status: Device set to", device)


# section: implementation
#---------------------------------------------------------------------------------a-----------------------------------
print("Execution status: Passed all prechecks, starting implementation")
model = MultiTaskModel(num_classes=5).to(device)
dummy_input = torch.randn(1, 3, 256, 256).to(device)
writer = SummaryWriter(log_dir="runs/multitask_experiment")
writer.add_graph(model, dummy_input)
print("\t step 1: Model initialized")
optimizer = optim.Adam(model.parameters(), lr=1e-4)
print("\t step 2: Optimizer initialized")
dataset = IDRiDDataset(image_paths, labels, mask_dirs, transform=transform)
val_dataset = IDRiDDataset(val_image_paths, val_labels, val_mask_dirs, transform=transform)
print("\t step 3: Dataset initialized")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
print("\t step 4: Data loader initiated")
num_epochs =5
print("\t step 5: Initializing training loop")
writer = SummaryWriter(log_dir="runs/multitask_training")
best_score = None
patience = 3
no_improve_count = 0
monitor_metric = 'accuracy'
for epoch in range(num_epochs):
    print(f"\t\tEpoch {epoch+1}/{num_epochs}: \n\t\t\tInitiating: Training Cycle")
    train(model, dataloader, optimizer, epoch, device)
    print(f"\t\t\tInitiating: Evaluating Cycle")
    acc, dice = evaluate(model, val_dataloader, device, epoch)
    print(f"\t\t\t\tAccuracy: {acc:.4f}, \n\t\t\t\tDice Score: {dice:.4f}")
    current_score = acc if monitor_metric == 'accuracy' else dice
    if best_score is None or current_score > best_score:
        best_score = current_score
        no_improve_count = 0
        save_model(model)
    else:
        no_improve_count += 1
        print(f"\t\t\tNo improvement for {no_improve_count} epoch(s).")
    if no_improve_count >= patience:
        print(f"\t\tEarly stopping triggered at epoch {epoch+1}.")
        break
print("Execution status: saving model")
save_model(model)
writer.close()
print("Execution status: Program completed successfully")