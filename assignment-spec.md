# Assignment-10: Custom U-Net lane segmentation from scratch

**Updated user instruction:** Use 65:35 (650 training / 350 validation images), seed 42, as the primary split. Retrain every layer from random initialization and regenerate all evaluation, memory, snapshot and submission artifacts. Keep the working dataset under `data/dataset_seg/`. Remove old experiments and redundant copies as requested; the original source manifests remain inside the working dataset as provenance.

Build a complete, runnable repository in a new working directory for the assignment described below. Use the existing PSU-reservoir polygon dataset from Assignment-8. Inspect the dataset and local hardware before choosing implementation details. Preserve the source dataset.

## Scope and source decisions

- Implement only single-class lane semantic segmentation: lane foreground versus background.
- Design a lightweight Custom U-Net yourself and randomly initialize every layer. Train all layers from scratch; do not use pretrained weights or a pretrained backbone.
- Use only polygon annotations. Bounding boxes and polylines are outside the assignment scope. A custom YOLO-seg model is not required by the supplied Brief.
- The lecture slides specify input H=36, W=64, C=3, batch size 2-4, at least 30 epochs, BCE plus Dice loss, and detection-positive IoU > 0.5. Use these as defaults.
- The unfinished text prompt uses 48x48 and IoU > 0.6. The handwritten notes use a 65:35 split and IoU >= 0.6. These conflict with the slides and existing dataset split. Document the conflicts; do not claim that all documents agree. Make input dimensions and evaluation IoU threshold configurable, and additionally report the >0.6 and >=0.6 variants if useful for comparison.
- The source dataset provides 800 training and 200 validation images. Preserve these original manifests and generate separate reproducible 65:35 manifests. Use the new 65:35 manifests as defaults in training, inference, evaluation, memory measurement and reporting. Do not describe validation results as independent test results. Inspect temporal proximity between frames and disclose possible leakage from the video-based split.

## 1. Dataset preparation

- Inspect `dataset_seg/data_seg.yaml`, image files, polygon labels, and train/validation manifests. Resolve relative paths robustly.
- Identify the class that represents the lane surface using the YAML names, DATACARD, and visual inspection. The current YAML declares `lane` as class ID 3; verify before using it.
- Rasterize all polygons of the selected lane class into one binary foreground mask. Treat other classes and unannotated pixels as background; do not merge all polygon classes into the lane class.
- Validate image-label pairing, class IDs, normalized polygon coordinates, and polygon validity. Handle empty lane masks explicitly and report malformed or missing annotations.
- Support source images of different sizes, including 1280x720. Resize RGB images to the configurable model input size and masks using nearest-neighbor interpolation. Apply spatial augmentation consistently to image and mask.
- Keep original image dimensions so inference can save masks at source resolution. Explain that accepting a source image is distinct from running the network directly at source resolution.

## 2. Custom network

- Create `model.py` implementing a small U-Net with an encoder, decoder, skip connections, and a one-channel output of logits.
- Choose depth and channel counts to fit local RAM/VRAM. Handle input dimensions such as 36x64 without skip-connection shape errors.
- Create `custom-unet-architecture.md` with a Mermaid architecture diagram, tensor dimensions for the default input, parameter count, and the rationale for the design.

## 3. Training

- Create `train.py` with the dataset preparation helpers or import them from a clearly organized dataset module.
- Use batch size 2-4 based on hardware and train for at least 30 epochs. Choose and document the optimizer and learning-rate scheduler.
- Combine binary cross-entropy on logits with Dice loss; document the weights and empty-mask handling.
- Randomly apply mild white-balance shifts, brightness shifts, and blur to training images. Do not apply photometric transforms to masks or random training augmentation during evaluation.
- Integrate TensorBoard for training and validation loss. Also save machine-readable training history and a loss plot.
- Save the best checkpoint selected using validation performance, plus the configuration, random seed, and class mapping required to reproduce inference.
- Run real training on the local machine. Record device, duration, epoch count, and results. Never invent training history or metrics. If training cannot finish, clearly identify the incomplete requirement.

## 4. Inference

- Create `inference.py` to load the trained checkpoint and process images without requiring annotation files.
- Convert predicted probabilities to binary masks using a documented configurable probability threshold. Keep this distinct from the evaluation IoU threshold.
- Save binary masks under `run/` in the new repository, restored to each source image's dimensions using nearest-neighbor interpolation. Use a documented representation such as 0/255 PNG.
- Generate at least one before/after snapshot showing the original image and the predicted lane overlay; also include ground truth when available.

## 5. Evaluation

- Create `evaluation.py` to load saved result masks and compare them with polygon-derived ground-truth masks at matching dimensions.
- Report pixel-wise lane IoU per image, mean IoU over all evaluated images, detection Yes/No using IoU > 0.5 by default, detection rate, and average IoU over detected images. Define behavior when there are no detected images or both masks are empty.
- Generate single-class AP and a precision-recall plot. Explicitly define the evaluation unit, confidence score, ranking or threshold sweep, IoU matching rule, and handling of images with no lane ground truth. Binary masks alone do not provide a confidence ranking: retain the probability maps or prediction confidence metadata needed for AP.
- If reporting pixel-level AP as an additional segmentation metric, name it separately from image/mask-level AP at an IoU threshold. Do not substitute one for the other without explanation.
- Save machine-readable metrics and plots. Use the same frozen evaluation split for reported comparisons.

## 6. Inference memory footprint

- Measure actual inference memory use on the available device; do not equate checkpoint file size with inference memory.
- Report process RAM baseline and peak, and CUDA allocated/reserved peak VRAM when running on CUDA. Explain the measurement method and its limitations.
- Include device, model input dimensions, batch size, precision, warm-up procedure, and whether preprocessing/postprocessing is included.
- Also report parameter count and checkpoint size as separate supporting information.

## 7. Submission and README

- Produce a self-contained repository with `model.py`, `train.py`, `inference.py`, `evaluation.py`, dependency instructions, architecture documentation, trained checkpoint or download instructions, configuration, and measured result artifacts.
- Write `README.md` explaining the network and design rationale, dataset selection and split, polygon-to-mask conversion, and commands for training, inference, evaluation, and memory measurement.
- Embed the actual training/validation loss plot, measured performance and precision-recall plot, at least one before/after inference snapshot, and actual inference memory footprint.
- Document the conflicting source requirements and the defaults adopted. Include limitations supported by the observed results.
- Prepare the repository for submission via a GitHub or PSU Storage URL according to the Brief. The lecture slides mention MS Teams as the submission channel. Do not upload, publish, or submit on the user's behalf unless requested.
- Verify the complete pipeline with real data and report what was executed and which requirements remain incomplete, if any.
