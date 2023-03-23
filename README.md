---
license: openrail
library_name: diffusers
---

This is the pretrained weights and some other detector weights of ControlNet.

See also: https://github.com/lllyasviel/ControlNet

# Description of Files

ControlNet/models/control_sd15_canny.pth

- The ControlNet+SD1.5 model to control SD using canny edge detection.

ControlNet/models/control_sd15_depth.pth

- The ControlNet+SD1.5 model to control SD using Midas depth estimation.

ControlNet/models/control_sd15_hed.pth

- The ControlNet+SD1.5 model to control SD using HED edge detection (soft edge).

ControlNet/models/control_sd15_mlsd.pth

- The ControlNet+SD1.5 model to control SD using M-LSD line detection (will also work with traditional Hough transform).

ControlNet/models/control_sd15_normal.pth

- The ControlNet+SD1.5 model to control SD using normal map. Best to use the normal map generated by that Gradio app. Other normal maps may also work as long as the direction is correct (left looks red, right looks blue, up looks green, down looks purple). 

ControlNet/models/control_sd15_openpose.pth

- The ControlNet+SD1.5 model to control SD using OpenPose pose detection. Directly manipulating pose skeleton should also work.

ControlNet/models/control_sd15_scribble.pth

- The ControlNet+SD1.5 model to control SD using human scribbles. The model is trained with boundary edges with very strong data augmentation to simulate boundary lines similar to that drawn by human.

ControlNet/models/control_sd15_seg.pth

- The ControlNet+SD1.5 model to control SD using semantic segmentation. The protocol is ADE20k.

ControlNet/annotator/ckpts/body_pose_model.pth

- Third-party model: Openpose’s pose detection model.

ControlNet/annotator/ckpts/hand_pose_model.pth

- Third-party model: Openpose’s hand detection model.

ControlNet/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt

- Third-party model: Midas depth estimation model.

ControlNet/annotator/ckpts/mlsd_large_512_fp32.pth

- Third-party model: M-LSD detection model.

ControlNet/annotator/ckpts/mlsd_tiny_512_fp32.pth

- Third-party model: M-LSD’s another smaller detection model (we do not use this one).

ControlNet/annotator/ckpts/network-bsds500.pth

- Third-party model: HED boundary detection.

ControlNet/annotator/ckpts/upernet_global_small.pth

- Third-party model: Uniformer semantic segmentation.

ControlNet/training/fill50k.zip

- The data for our training tutorial.

# Related Resources

Special Thank to the great project - [Mikubill' A1111 Webui Plugin](https://github.com/Mikubill/sd-webui-controlnet) !

We also thank Hysts for making [Gradio](https://github.com/gradio-app/gradio) demo in [Hugging Face Space](https://huggingface.co/spaces/hysts/ControlNet) as well as more than 65 models in that amazing [Colab list](https://github.com/camenduru/controlnet-colab)! 

Thank haofanwang for making [ControlNet-for-Diffusers](https://github.com/haofanwang/ControlNet-for-Diffusers)!

We also thank all authors for making Controlnet DEMOs, including but not limited to [fffiloni](https://huggingface.co/spaces/fffiloni/ControlNet-Video), [other-model](https://huggingface.co/spaces/hysts/ControlNet-with-other-models), [ThereforeGames](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/7784), [RamAnanth1](https://huggingface.co/spaces/RamAnanth1/ControlNet), etc!

# Misuse, Malicious Use, and Out-of-Scope Use

The model should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.