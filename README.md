# Part 1
[[Paper]]()

Face generators based on [StyleGAN2](https://github.com/NVlabs/stylegan2.git)
#### References
* [InterFaceGAN - Interpreting the Latent Space of GANs for Semantic Face Editing](https://github.com/genforce/interfacegan.git)
* [Generators-with-stylegan2](https://github.com/a312863063/generators-with-stylegan2.git)

### Requirements
* Python 3.6 is used. 

```
pip3 install tensorflow==1.14
pip3 install tensorflow-gpu==1.14
pip3 install numpy==1.16.4
```

### Usage
1. Download the model [stylegan2-ffhq-config-f.pkl](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/) to a folder /networks 
2. Generate image
```
python3 main.py
```
```
python3 create_photo.py
```
<img src='./results/0.png' width=200> <img src='./results/1.png' width=200> <img src='./results/2.png' width=200> <img src='./results/3.png' width=200>  

3. 
We can select the necessary functions to change the image using the notebook.

[Colab]()

<img src='./results/screen/screen_15.png' width=250 height=250> <img src='./results/param/15.png' width=250>

<img src='./results/screen/screen_36.png' width=250 height=250> <img src='./results/param/36.png' width=250>

<img src='./results/screen/screen_73.png' width=250 height=250> <img src='./results/param/88.png' width=250>

<img src='./results/screen/screen_91.png' width=250 height=250> <img src='./results/param/91.png' width=250>

OR 

We can apply all latent identifiers to the selected generated image. 

```
python3 photo_generated.py
```

age 

<img src='./results/age/000.png' width=200> <img src='./results/age/005.png' width=200> <img src='./results/age/007.png' width=200> <img src='./results/age/008.png' width=200>

gender

<img src='./results/gender/000.png' width=200>  <img src='./results/gender/005.png' width=200> <img src='./results/gender/007.png' width=200> <img src='./results/gender/008.png' width=200>

emotion

<img src='./results/emotion_sad/000.png' width=200>  <img src='./results/emotion_sad/005.png' width=200> <img src='./results/emotion_sad/007.png' width=200> <img src='./results/emotion_sad/008.png' width=200>



## Additional material
#### Pre-trained networks

1. #### [networks](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/)

- StyleGAN2 for FFHQdataset at 1024&times;1024 (stylegan2-ffhq-config-f.pkl)

2. #### [latent vectors](https://github.com/a312863063/generators-with-stylegan2)
- age.npy
- angle_horizontal.npy
- angle_vertical.npy
- beauty.npy
- emotion_angry.npy 
- emotion_disgust.npy
- emotion_easy.npy 
- emotion_fear.npy 
- emotion_happy.npy 
- emotion_sad.npy 
- emotion_surprise.npy
- eyes_open.npy 
- face_shape.npy 
- gender.npy 
- glasses.npy 
- height.npy 
- race_black.npy 
- race_white.npy
- race_yellow.npy 
- smile.npy 
- width.npy



# Part 2
### Paper:
[[Rotate-and-Render: Unsupervised Photorealistic Face Rotation from Single-View Images]](https://arxiv.org/abs/2003.08124)

[[Morphable Model For The Synthesis Of 3D Faces]](https://www.face-rec.org/algorithms/3d_morph/morphmod2.pdf)
#### References
[Rotate-and-Render: Unsupervised Photorealistic Face Rotation from Single-View Images (CVPR 2020)](https://github.com/Hangz-nju-cuhk/Rotate-and-Render)

[face3d: Python tools for processing 3D face](https://github.com/YadiraF/face3d.git)
### Requirements
* Python 3.6 is used.


### Usage

1.
```
git clone https://github.com/YadiraF/face3d
```
2. 
Prepare BFM Data
[Data/BFM/readme.md](https://github.com/YadiraF/face3d/blob/master/examples/Data/BFM/readme.md)

3. 
Write the name of the photo to be generated to a file file_list.txt

4.
Run generate_image_map.py
```bash
python generate_image_map.py 
```
#### Results 
 
<img src='./results_2/16.png' width=200> <img src='./results_2/2.png' width=200>  <img src='./results_2/10.png' width=200> <img src='./results_2/3.png' width=200> 
