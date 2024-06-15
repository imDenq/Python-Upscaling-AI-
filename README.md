Video Super-Resolution with SRCNN
Description
This project provides a script to perform video super-resolution using a Super-Resolution Convolutional Neural Network (SRCNN). It extracts frames from low-quality (LQ) videos, applies super-resolution to enhance their quality, and then reconstructs the enhanced frames back into high-quality (HQ) videos.

Key Features
Efficient Frame Extraction: Multi-threaded frame extraction from videos.
Deep Learning Model: SRCNN model optimized with mixed precision training.
Parallel Processing: Utilizes GPU for fast super-resolution inference.
Reconstruction: Combines enhanced frames back into videos.
System Requirements
Operating System: Linux, Windows, or macOS.
Python: Version 3.7 or higher.
PyTorch: Version compatible with CUDA 11.8 or higher.
CUDA: Required for GPU acceleration.
Hardware: NVIDIA GPU (RTX 4090 recommended for optimal performance).
Installation
Clone the Repository:

bash
Copier le code
git clone https://github.com/your-username/video-super-resolution.git
cd video-super-resolution
Set Up the Python Environment:

Create a virtual environment (optional but recommended):
bash
Copier le code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install Dependencies:

Install the required packages:
bash
Copier le code
pip install -r requirements.txt
Note: Ensure you have the appropriate version of PyTorch for your CUDA installation. For example, for CUDA 11.8:
bash
Copier le code
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
Usage
Prepare the Video Data:

Place your high-quality (HQ) and low-quality (LQ) videos in hq_videos and lq_videos directories respectively.
Adjust the paths in the script if your directory structure is different.
Run the Script:

Execute the script to extract frames, train the SRCNN model, apply super-resolution, and combine frames back into videos:
bash
Copier le code
python video_super_resolution.py
Output:

The processed frames will be saved in upscaled_frames directory.
The final upscaled videos will be saved in upscaled_videos directory.
Script Parameters
Frame Rate: Adjust the frame_rate parameter in the extract_frames_optimized function to control the extraction frequency.
Batch Size: Modify the batch_size in the train_model_optimized and upscale_video_optimized functions to optimize GPU utilization based on your hardware.
Example Usage
Modify the paths and filenames in the hq_video_paths and lq_video_paths lists as per your input video files:

python
Copier le code
hq_video_paths = [
    'hq_videos/4.mp4',
    # Add more HQ video paths as needed
]

lq_video_paths = [
    'lq_videos/3.mp4',
    # Add more LQ video paths as needed
]
Performance Optimization
GPU Utilization: Ensure that CUDA and cuDNN are properly installed and configured.
Batch Processing: Use larger batch sizes to fully utilize GPU memory.
Data Parallelism: Increase the number of workers for data loading to speed up processing.
Troubleshooting
Low GPU Utilization: Verify that the GPU is correctly detected and being used by PyTorch. Check the CUDA version compatibility.
Out of Memory Errors: Reduce the batch size or optimize the DataLoader settings.
Contribution
Contributions are welcome! Feel free to submit a pull request or open an issue to discuss improvements or fixes.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgements
This project is based on the SRCNN model for image super-resolution. Special thanks to the deep learning community and open-source contributors for their valuable resources and support.
