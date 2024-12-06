# Image-to-Image Colorization with Stable Diffusion  

This project demonstrates a pipeline for fine-tuning a **Stable Diffusion** model bk-sdm-small to perform **image-to-image colorization**. It transforms grayscale images into colorized versions. The mentioned model is efficient and light weight. 
---
## About the Lightweight Stable diffusion model
Used this model [bk-sdm-small](https://huggingface.co/nota-ai/bk-sdm-small)

Stable Diffusion models (SDMs) involves high computing demands due to billion-scale parameters.
To enhance efficiency, recent studies have reduced sampling steps and
applied network quantization while retaining the original architectures.
The lack of architectural reduction attempts may stem from worries over
expensive retraining for such massive models. In this work, we uncover the
surprising potential of block pruning and feature distillation for low-cost
general-purpose T2I. By removing several residual and attention blocks
from the U-Net of SDMs, we achieve 30% ∼50% reduction in model size,
MACs, and latency. We show that distillation retraining is effective even
under limited resources: using only 13 A100 days and a tiny dataset, our
compact models can imitate the original SDMs (v1.4 and v2.1-base with
over 6,000 A100 days). Benefiting from the transferred knowledge, our BKSDMs deliver competitive results on zero-shot MS-COCO against larger
multi-billion parameter models. We further demonstrate the applicability
of our lightweight backbones in personalized generation and image-toimage translation. Deployment of our models on edge devices attains
4-second inference
---
## About the dataset
The dataset is available here
Dataset processing
- Collected 150 samples from unsplash. All images are license free.
- Converted the images to Grayscale(Black and white color)
- Organize the dataset into two directories:  
- **`input/`**: Contains grayscale images.  
- **`output/`**: Contains the corresponding colorized images. 
---
## Key Features  
- **Data Augmentation**: Includes random transformations like rotation, cropping, and jittering for robust training.    
- **Custom Dataset Loader**: Easily adapt the pipeline to any dataset of grayscale and color image pairs.
- **Fine-Tuning Stable Diffusion**: Customize the UNet in Stable Diffusion for colorization tasks.
- **Image-to-Image Pipeline**: Uses grayscale images as input and generates corresponding colorized outputs.   
---

## Future Improvements  
- Add support for additional tasks (e.g., sketch-to-image generation).  
- Train the dataset with more large and diverse dataset.    
---
## Acknowledgments  
- [Hugging Face Diffusers Library](https://huggingface.co/docs/diffusers/)  
- [PyTorch](https://pytorch.org/)  
- [VGG-19 for Perceptual Loss](https://pytorch.org/vision/stable/models.html#torchvision.models.vgg19)

---

## License  
This project is licensed under the MIT License.  

---  
Feel free to customize the description based on your specific needs or goals!
