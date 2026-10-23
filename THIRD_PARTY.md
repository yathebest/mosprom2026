# Sources and third-party components

## YOLO12m and expanded PPE training data

| Component | Primary source | Notes |
|---|---|---|
| YOLO12m COCO weights | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12m.pt | Ultralytics AGPL-3.0; local downloaded checkpoint, SHA-256 in model manifest |
| Construction Site Safety | https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow | Source README declares CC BY 4.0; retain upstream source list and attribution |
| SHWD | https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset | MIT repository, includes SCUT-HEAD-derived imagery; preserve original notices. `person` means bare head |
| HardHat-Vest v3 | https://www.kaggle.com/datasets/muhammetzahitaydn/hardhat-vest-dataset-v3 | Kaggle declares CC0 but aggregates several upstream corpora; this is not a blanket rights audit. Used locally for explicitly labelled positive crops only |
| ResNet50/152 initialization | https://github.com/pytorch/vision | Official torchvision pretrained weights; trained classifier artifacts are not yet supplied |
| Optional CUDA ONNX Runtime | https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html | MIT; isolated ORT GPU 1.20.2 CUDA 12 / cuDNN 9 profile |

Raw archives and generated datasets remain outside Git. The private training ZIP retains source notices, provenance JSON and archive hashes. Exact duplicates/near duplicates and known source groups are handled conservatively; participant/camera independence is not established. LinQ Framework SDK was supplied privately by the user and is not redistributed in Git.

## Added scenario models and samples

| Component | Primary source | Attribution / terms |
|---|---|---|
| PPE13 ONNX | https://huggingface.co/ayushgupta7777/safetyvision-yolov8 ; https://github.com/ayushgupta07xx/SafetyVision | Ayush Gupta; AGPL-3.0; negative PPE classes; fall class ignored |
| Fire/smoke | https://github.com/luminous0219/fire-and-smoke-detection-yolov8 | Upstream AGPL-3.0; restricted checkpoint loading and ONNX conversion |
| Smoking ONNX | https://github.com/alihassanml/Smoking-detection-yolo11 | Ali Hassan; repository MIT, ONNX metadata declares Ultralytics AGPL-3.0. Preserve both notices; not MIT-only |
| Construction/machinery | https://huggingface.co/baskarmother/yolov8-ppe-construction | Model card MIT; underlying Ultralytics AGPL. Actual class names retained |
| Phone detector | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt | Official YOLOv8s / COCO, AGPL-3.0; larger than carried-object nano model; full-frame + person crops |
| COCO YOLOv8n | https://github.com/ultralytics/assets/releases | Official Ultralytics weights, AGPL-3.0; carried objects |
| ONNX Runtime | https://github.com/microsoft/onnxruntime | MIT; local CPU execution |
| Construction-PPE images | https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip ; https://docs.ultralytics.com/datasets/detect/construction-ppe/ | Official distribution; retain original dataset terms. Underlying photographs may have different rights; software license is not a blanket image license |
| COCO128 images | https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip ; https://cocodataset.org/ | Images retain original licenses; technical fixtures, training overlap possible |
| Fire / smoking examples | Same repositories above | Author demonstrations; smoking inputs include author annotations. Repeated-frame videos and evidence annotations made locally |

Pinned revisions, URLs and SHA256 are in data/models/scenario-sources.json and scenario-manifest.json. docs/scenario_exports.json records conversions. No upstream application source was copied. Author metrics are not our results. Original scenario checkpoints are excluded from the portable archive; deployed ONNX and attribution are included.

The application-specific code was authored for this task. Existing libraries and pretrained models are reused through their APIs.

| Component | Source | Terms / purpose |
|---|---|---|
| Ultralytics YOLO / bundled BYTETracker implementation | https://github.com/ultralytics/ultralytics | AGPL-3.0; model inference and tracking |
| YOLOv8n-pose checkpoint | https://github.com/ultralytics/assets/releases | Ultralytics model; downloaded official asset, hash in data/models/manifest.json |
| PyTorch | https://github.com/pytorch/pytorch | BSD-style; tensor execution |
| OpenCV | https://github.com/opencv/opencv | Apache-2.0 for current OpenCV releases; decode and geometry |
| FastAPI | https://github.com/fastapi/fastapi | MIT; local API |
| imageio-ffmpeg / FFmpeg | https://github.com/imageio/imageio-ffmpeg | Wrapper BSD-2-Clause; bundled FFmpeg has its own build-specific licenses |
| UR Fall Detection | https://fenix.ur.edu.pl/~mkepski/ds/uf.html | CC BY-NC-SA 4.0; evaluation samples |
| SH17 | https://github.com/ahmadmughees/SH17dataset | CC BY-NC-SA 4.0; proposed optional PPE dataset, not bundled |

URFD attribution: Bogdan Kwolek and Michal Kepski, *Human fall detection on embedded platform using depth maps and wireless accelerometer*, Computer Methods and Programs in Biomedicine 117(3), 2014, pp. 489–501. The preview videos were cropped to retain the RGB half and re-encoded; transformations and source hashes are documented in their manifests. Preserve attribution and dataset terms when redistributing derived samples.

Design/research references:

- https://github.com/nisha572/ai-construction-safety-detection — studied pipeline separation and temporal rules; README declares MIT. Application source was not copied.
- https://github.com/nusRying/ai-safety-monitoring — feature comparison only; proprietary/all rights reserved.
- https://github.com/IlliaSator/industrial-safety-vision — architecture comparison; no explicit license was found in the inspected checkout, so its source was not reused.
- https://github.com/roboflow/supervision — MIT; optional future utility library.
- https://www.intenseye.com/ and https://www.protex.ai/ — product comparisons only; no vendor software is included.

These are research examples, not endorsements or claims of equivalent accuracy. Reported vendor/repository results were not reproduced here.


ResNet18 architecture and ImageNet initialization: official torchvision [resnet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), [torchvision source](https://github.com/pytorch/vision). The included `fall-resnet18.pt` is a task-specific fine-tuned checkpoint, trained on the URFD RGB training split; its manifest records the hash and provenance. The original ImageNet initialization cache is not included in the portable archive. Retain the upstream software licenses and dataset attribution; do not infer unrestricted commercial data rights from the model architecture license.
