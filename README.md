# fps_upscaling
 I am practicing here to try to reproduce the content of this article https://arxiv.org/abs/1706.01159

to do:
	- replace the resnet18-discriminator by a "LeNet-like architecture with 16 layers"
	- Incorporating optical flow prior approach

---

Training on Sintel (240p). Testing on Spring (240p).


MSE approach (epoch=10):
mse: 0.0008294550174261431
psnr: 34.76865184866296
ssim :0.9559021068303339
GAN approach (epoch=10, gamma=0.5):
mse: 0.0008051188071697995
psnr: 35.35519524055286
ssim :0.9553437313848879

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
	- I train and test on 240p videos. (They do not specify the definition in the paper, a priori it is full HD.)
	- I use currently a resnet-18 as a discriminator and not a "LeNet-like architecture with 16 layers"