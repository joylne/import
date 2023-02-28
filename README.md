---
license: openrail
datasets:
- HighCWu/fill50k
pipeline_tag: text-to-image
---


# Model Card for ControlNet

<!-- Provide a quick summary of what the model is/does. -->
ControlNet is a neural network structure to control diffusion models by adding extra conditions.

*This is the pretrained weights and some other detector weights of ControlNet.*

## Model Details

### Model Description

The [associated paper](https://arxiv.org/pdf/2302.05543.pdf) details: 
> ControlNet, an end-to-end neural network architecture that controls large image diffusion models (like Stable Diffusion) to learn task-specific input conditions.


- **Developed by:**  Lvmin Zhang and Maneesh Agrawala
- **Shared by:** [Lvmin Zhang](https://huggingface.co/lllyasviel)
- **Model type:** Text to Image
- **Language(s) (NLP):** [More Information Needed]
- **License:** OpeRAIL
- **Finetuned from model [optional]:** [More Information Needed]

### Model Sources]

<!-- Provide the basic links for the model. -->

- **Repository:** [Github](https://github.com/lllyasviel/ControlNet)
- **Paper:** [Paper](https://arxiv.org/abs/2302.05543)
- **Demo:** [Hugging Face Space](https://huggingface.co/spaces/hysts/ControlNet)

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->
Control pretrained large diffusion models to support additional input conditions. 

### Downstream Use

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

See [Description of Files](####Software) section for different models to control Stable Diffusion.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

The model should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

[More Information Needed]

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

### How to Get Started with the Model

Use the code below to get started with the model.

[More Information Needed]

## Training Details

### Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

See [fill50k Dataset Card](https://huggingface.co/datasets/HighCWu/fill50k) and the ControlNet/training/fill50k.zip file for the data for our training tutorial.

### Training Procedure 

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

[More Information Needed]


#### Training Hyperparameters

- **Training regime:** [More Information Needed] <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Data Card if possible. -->

[More Information Needed]

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

[More Information Needed]

### Results

[More Information Needed]

#### Summary

[More Information Needed]

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** Nvidia A100 80G
- **Hours used:** 600
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

The [associated paper](https://arxiv.org/pdf/2302.05543.pdf) detials the compute hardware and GPU hours for:
> several implementations of ControlNets with different image-based conditions to control large diffusion models in various ways.
The above hardware and GPU hours corresponds to the **Canny Edge detector**

## Technical Specifications

### Model Architecture and Objective
As noted in the [associated Github](https://github.com/lllyasviel/ControlNet):

>It copys the weights of neural network blocks into a "locked" copy and a "trainable" copy.
>The "trainable" one learns your condition. The "locked" one preserves your model.
>Thanks to this, training with small dataset of image pairs will not destroy the production-ready diffusion models.
>The "zero convolution" is 1×1 convolution with both weight and bias initialized as zeros.
>Before training, all zero convolutions output zeros, and ControlNet will not cause any distortion.
>No layer is trained from scratch. You are still fine-tuning. Your original model is safe.
>This allows training on small-scale or even personal devices.>


### Compute Infrastructure

[More Information Needed]

#### Hardware

[More Information Needed]

#### Software

**Description of Files**

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



## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->


**BibTeX:**
```
@misc{https://doi.org/10.48550/arxiv.2302.05543,
  doi = {10.48550/ARXIV.2302.05543},
  
  url = {https://arxiv.org/abs/2302.05543},
  
  author = {Zhang, Lvmin and Agrawala, Maneesh},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), Graphics (cs.GR), Human-Computer Interaction (cs.HC), Multimedia (cs.MM), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Adding Conditional Control to Text-to-Image Diffusion Models},
  
  publisher = {arXiv},
  
  year = {2023},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```


## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information

Special Thank to the great project - [Mikubill' A1111 Webui Plugin](https://github.com/Mikubill/sd-webui-controlnet) !

We also thank Hysts for making [Gradio](https://github.com/gradio-app/gradio) demo in [Hugging Face Space](https://huggingface.co/spaces/hysts/ControlNet) as well as more than 65 models in that amazing [Colab list](https://github.com/camenduru/controlnet-colab)! 

Thank haofanwang for making [ControlNet-for-Diffusers](https://github.com/haofanwang/ControlNet-for-Diffusers)!

We also thank all authors for making Controlnet DEMOs, including but not limited to [fffiloni](https://huggingface.co/spaces/fffiloni/ControlNet-Video), [other-model](https://huggingface.co/spaces/hysts/ControlNet-with-other-models), [ThereforeGames](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/7784), [RamAnanth1](https://huggingface.co/spaces/RamAnanth1/ControlNet), etc!
## Model Card Authors

[Lvmin Zhang](https://huggingface.co/lllyasviel) in collaboration with [Ezi Ozoani](https://huggingface.co/Ezi) and the Hugging Face team.

## Model Card Contact

[More Information Needed]

