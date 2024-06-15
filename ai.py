import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.transforms import ToTensor, Grayscale, Compose
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from torch.cuda.amp import GradScaler, autocast

# Define SRCNN Model
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Optimized function to extract frames using threading
def extract_frames_optimized(video_path, output_folder, frame_rate=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    index = 0

    def process_frame(frame, index):
        frame_filename = os.path.join(output_folder, f"frame_{index:06d}.png")
        cv2.imwrite(frame_filename, frame)

    with ThreadPoolExecutor(max_workers=12) as executor:  # Using more threads for your CPU
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if index % frame_rate == 0:
                executor.submit(process_frame, frame, index)
            index += 1
    cap.release()

# Custom Dataset for HQ-LQ frames with optimized transformations
class VideoSuperResolutionDataset(Dataset):
    def __init__(self, hq_folder, lq_folder):
        self.hq_files = sorted([os.path.join(hq_folder, f) for f in os.listdir(hq_folder) if f.endswith('.png')])
        self.lq_files = sorted([os.path.join(lq_folder, f) for f in os.listdir(lq_folder) if f.endswith('.png')])
        self.transform = Compose([Grayscale(), ToTensor()])

    def __len__(self):
        return len(self.hq_files)

    def __getitem__(self, idx):
        hq_image = Image.open(self.hq_files[idx])
        lq_image = Image.open(self.lq_files[idx])
        
        hq_image = self.transform(hq_image)
        lq_image = self.transform(lq_image)

        return lq_image, hq_image

# Function to prepare HQ and LQ frames for videos
def prepare_video_frames(hq_video_paths, lq_video_paths, hq_frames_folder, lq_frames_folder):
    for hq_video, lq_video in zip(hq_video_paths, lq_video_paths):
        video_name = os.path.basename(hq_video).split('.')[0]
        hq_output_folder = os.path.join(hq_frames_folder, video_name)
        lq_output_folder = os.path.join(lq_frames_folder, video_name)
        
        extract_frames_optimized(hq_video, hq_output_folder, frame_rate=30)
        extract_frames_optimized(lq_video, lq_output_folder, frame_rate=30)

# Optimized training function with mixed precision
def train_model_optimized(model, dataloader, num_epochs=100, learning_rate=0.001, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, targets = data
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            with autocast():  # Enable mixed precision
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            if i % 10 == 9:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0

    print('Training completed')
    return model

# Optimized function to upscale video frames using threading
def upscale_video_optimized(model, lq_folder, output_folder, batch_size=32):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    model.eval()
    lq_files = [f for f in os.listdir(lq_folder) if f.endswith('.png')]
    lq_files.sort()
    
    def load_and_process(files):
        batch = []
        for file in files:
            lq_image = Image.open(os.path.join(lq_folder, file)).convert('L')
            lq_tensor = ToTensor()(lq_image).unsqueeze(0)
            batch.append(lq_tensor)
        return torch.cat(batch).cuda()

    for i in range(0, len(lq_files), batch_size):
        batch_files = lq_files[i:i + batch_size]
        lq_tensors = load_and_process(batch_files)
        
        with torch.no_grad():
            sr_tensors = model(lq_tensors)
        
        for j, file in enumerate(batch_files):
            sr_image = sr_tensors[j].squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0
            sr_image = sr_image.clip(0, 255).astype('uint8')
            sr_image = Image.fromarray(sr_image)
            sr_image.save(os.path.join(output_folder, file))

# Function to combine upscaled frames into a video
def combine_frames_to_video_optimized(frames_folder, output_video, frame_rate=30):
    frames = sorted([f for f in os.listdir(frames_folder) if f.endswith('.png')])
    first_frame = cv2.imread(os.path.join(frames_folder, frames[0]))
    height, width, layers = first_frame.shape
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    def write_frame(frame_file):
        frame = cv2.imread(os.path.join(frames_folder, frame_file))
        video.write(frame)
    
    with ThreadPoolExecutor(max_workers=12) as executor:
        executor.map(write_frame, frames)

    video.release()

# Example usage
hq_video_paths = [
    'hq_videos/4.mp4',
    # Add more HQ video paths as needed
]

lq_video_paths = [
    'lq_videos/3.mp4',
    # Add more LQ video paths as needed
]

# Prepare HQ and LQ frames for videos
prepare_video_frames(hq_video_paths, lq_video_paths, 'hq_frames', 'lq_frames')

# Create DataLoader for the videos
hq_frame_folders = [os.path.join('hq_frames', f) for f in os.listdir('hq_frames')]
lq_frame_folders = [os.path.join('lq_frames', f) for f in os.listdir('lq_frames')]

datasets = []
for hq_folder, lq_folder in zip(hq_frame_folders, lq_frame_folders):
    dataset = VideoSuperResolutionDataset(hq_folder, lq_folder)
    datasets.append(dataset)

combined_dataset = ConcatDataset(datasets)
dataloader = DataLoader(
    combined_dataset,
    batch_size=32,  # Increased batch size for better GPU utilization
    shuffle=True,
    num_workers=12,  # Utilize 12 workers to leverage your CPU's cores
    pin_memory=True
)

# Initialize the model
model = SRCNN().cuda()

# Train the model
model = train_model_optimized(model, dataloader, num_epochs=100, learning_rate=0.001, batch_size=32)

# Save the trained model
torch.save(model.state_dict(), 'srcnn_video_super_resolution.pth')

# Apply the model to upscale video frames
for lq_folder in lq_frame_folders:
    video_name = os.path.basename(lq_folder)
    output_folder = os.path.join('upscaled_frames', video_name)
    upscale_video_optimized(model, lq_folder, output_folder, batch_size=32)

# Combine upscaled frames into videos
for output_folder in os.listdir('upscaled_frames'):
    combine_frames_to_video_optimized(os.path.join('upscaled_frames', output_folder), f'upscaled_videos/{output_folder}.mp4')
