

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image
import onnx

# 1. Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tif')])
        self.target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.tif')])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_image = ToTensor()(Image.open(self.input_files[idx]))
        target_image = ToTensor()(Image.open(self.target_files[idx]))
        return input_image, target_image

# 2. ResNet-like Neural Network
class SimpleResNet(nn.Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)
        return x + residual

# 3. Training
if __name__ == "__main__":
    # Directories
    input_dir = "E:\\Deeplearning\\C11800\\High\\N2N\\object_noise"
    target_dir = "E:\\Deeplearning\\C11800\\High\\N2N\\noise"

    # Parameters
    batch_size = 32
    epochs = 20
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset = CustomDataset(input_dir, target_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, Optimizer
    model = SimpleResNet().to(device)
    criterion = nn.L1Loss()  # MAE
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

    # Save Model as ONNX
    dummy_input = torch.randn(1, 1, 256, 256).to(device)
    torch.onnx.export(model, dummy_input, "resnet_model.onnx", opset_version=11)

    # 4. Inference
    model.eval()
    test_image_path = "path_to_test_image.tif"
    test_image = ToTensor()(Image.open(test_image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(test_image)

    output_image = output.squeeze().cpu().numpy()
    output_image = (output_image * 255).astype(np.uint8)
    Image.fromarray(output_image).save("output_image.tif")

print("Training and inference complete.")






































