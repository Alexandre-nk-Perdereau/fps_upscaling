# fps_upscaling
 I am practicing here to try to reproduce the content of this article https://arxiv.org/abs/1706.01159

to do:
	- determine the cause of the " flashing " effect on the video test
	- Incorporating optical flow prior approach

---

A/ test on image pairs
Training on Sintel (240p). Testing on Spring (240p).


MSE approach (epoch=30):
mse: 0.0008770605956757294
psnr: 34.05617756477495
ssim :0.9519715887451657
GAN approach (epoch=30, gamma=0.5):
mse: 000764338637911684
psnr: 35.97348879848563
ssim : 0.958871929578785

to compare, the result in the article:

MSE approach:
mse: 0.0050
psnr: 23
ssim :0.614
GAN approach:
mse: 0.0053
psnr: 22.8
ssim 0.721

difference between what is described in the paper and my work:
	- not the same test set, I used another blender movie to avoid bias.
	- I train and test on 240p videos. (They do not specify the definition in the paper, a priori they use a full HD video.)
	- I use currently a resnet-18 as a discriminator and not a "LeNet-like architecture with 16 layers"