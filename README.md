Welcome to the YOLOv8 Human Detection Beginner's Repository – your entry point into the exciting world of object detection! This repository is tailored for beginners, providing a straightforward implementation of YOLOv8 for human detection in images and videos.

## Features:

1. **User-Friendly Implementation:** Designed with simplicity in mind, this repository offers a beginner-friendly implementation of YOLOv8 for human detection. No advanced knowledge of deep learning or computer vision is required to get started.

2. **Pre-trained Model:** Start detecting humans right away with our pre-trained YOLOv8 model. The model has been trained on a variety of datasets, created and imported from Roboflow, to ensure versatility in different scenarios.

3. **Minimal Dependencies:** We've kept the dependencies minimal to make it easy for beginners to set up the environment and start using the human detection model without any hassle.

4. **Example Code:** Explore example code and scripts to understand how to integrate the YOLOv8 model into your own projects. The repository includes two Python notebooks:
   - `training.ipynb`: Use this notebook for training the YOLOv8 model on your custom datasets or additional data.
   - `predictions.ipynb`: Utilize this notebook for making predictions and running the trained model on new images or videos.

5. **Model Checkpoints:**
   - `best.pt`: This file contains the weights of the YOLOv8 model at the epoch where it achieved the best performance during training.
   - `last.pt`: Save time and continue training or fine-tuning the model from the point it left off using this file, which contains the weights from the last epoch.

6. **Roboflow Integration:** Easily create custom datasets for training by leveraging Roboflow. Use their platform to annotate images, manage datasets, and export the data in YOLOv8-compatible format, streamlining the process of preparing your own data for training.

7. **Dataset Specifications:**
   - **Dataset Split:**
      - TRAIN SET: 88%, 4200 Images
      - VALID SET: 8%, 400 Images
      - TEST SET: 4%, 200 Images

   - **Preprocessing:**
      - Auto-Orient: Applied
      - Resize: Stretch to 640x640
      - Auto-Adjust Contrast: Using Histogram Equalization

   - **Augmentations:**
      - Outputs per training example: 3
      - Flip: Horizontal
      - Rotation: Between -15° and +15°
      - Grayscale: Apply to 15% of images
      - Hue: Between -15° and +15°
      - Saturation: Between -15% and +15%
      - Brightness: Between -20% and +0%
      - Blur: Up to 2.5px

8. **Training Guide:**
   ### Step-by-Step Guide: Training YOLOv8 on Custom Human Detection Dataset

   #### 1. Install Required Libraries
   ```python
   # Install the Ultralytics library using pip
   !pip install ultralytics

   # Install the Roboflow library using pip
   !pip install roboflow
   ```

   #### 2. Import Libraries
   ```python
   # Import the Roboflow and YOLO modules
   from roboflow import Roboflow
   from ultralytics import YOLO

   # The 'Roboflow' module is used for working with datasets on the Roboflow platform
   # The 'YOLO' module is imported from the 'ultralytics' library, which is used for YOLO object detection tasks
   ```

   #### 3. Set Up Roboflow Integration
   ```python
   # Import the Roboflow library and create an instance with the provided API key
   api_key = "your-api-key"
   rf = Roboflow(api_key)

   # Access the Roboflow workspace named "lazydevs"
   # Access the project named "human-detection" within the workspace
   workspace_name = "lazydevs"
   project_name = "human-detection"
   project = rf.workspace(workspace_name).project(project_name)
   ```

   #### 4. Download Dataset from Roboflow
   ```python
   # Download the dataset associated with version 4 of the project using YOLOv8 format
   # Note: You might want to include a specific version number or method for version selection.
   version = 4
   form = "yolov8"
   dataset = project.version(version).download(form)
   ```

   #### 5. Initialize YOLOv8 Model
   ```python
   # Initialize the YOLO model by loading the pre-trained weights from 'yolov8n.pt'
   model = YOLO('yolov8n.pt')
   ```

   #### 6. Train the Model
   ```python
   # Train the model with the specified parameters
   results = model.train(
       data='/kaggle/input/v4-yaml/data.yaml',  # Path to the training data YAML file
       epochs=150,  # Number of training epochs
       batch=64,  # Batch size for training
       imgsz=640,  # Input image size
       seed=32,  # Random seed for reproducibility
       optimizer='NAdam',  # Optimizer algorithm
       weight_decay=1e-4,  # Weight decay for regularization
       momentum=0.937,  # Initial momentum for the optimizer
       cos_lr=True,  # Use cosine learning rate scheduling
       lr0=0.01,  # Initial learning rate
       lrf=1e-5,  # Final learning rate
       warmup_epochs=10,  # Number of warmup epochs
       warmup_momentum=0.5,  # Momentum during warm-up epochs
       close_mosaic=20,  # Parameter for close mosaic augmentation
       label_smoothing=0.2,  # Label smoothing parameter for regularization
       dropout=0.5,  # Dropout rate to prevent overfitting
       verbose=True  # Print verbose training information
   )
   ```

Follow these steps to train the YOLOv8 model on your custom human detection dataset. Adjust parameters and paths according to your specific requirements.

9. **Documentation for Beginners:** The documentation provides clear and concise instructions on setting up the environment, running the model, and understanding the basics of YOLOv8 for human detection. Perfect for those taking their first steps into the world of computer vision.

## Getting Started:

1. Clone the repository: `git clone https://github.com/J3lly-Been/YOLOv8-HumanDetection.git`
2. Follow the beginner-friendly setup instructions in the documentation to install necessary dependencies and get the model up and running.
3. Explore the example code to understand how to use the pre-trained YOLOv8 model for human detection and leverage the provided notebooks for training and predictions. Additionally, use `best.pt` and `last.pt` for different scenarios, such as starting from the best-performing weights or continuing training.
4. Utilize Roboflow to create custom datasets, annotate images, and seamlessly integrate your own data into the YOLOv8 model training process.

## Contributions:

This repository is open to contributions from the beginner community. If you have ideas to improve simplicity, clarity, or add new features suitable for beginners, feel free to submit your contributions through issues or pull requests.

Embark on your journey into human detection with YOLOv8 using this beginner-friendly repository. Start detecting humans in no time and gain hands-on experience with object detection!
