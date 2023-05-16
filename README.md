# PVT_v2_video_frame_interpolation
使用PVT_v2作为编码器的视频插帧程序，A program using PVT_v2 as the encoder of video frame interpolation, VFI, pytorch

## 总体情况

如果你是初学者，本项目下载之后即可直接运行，无需了解相关架构，仅需简单python/pytorch基础，阅读文件说明，如果有问题可以通过邮箱联系

本代码实现了通过前后两帧预测中间帧的任务，使用Encoder-Decoder架构。

本README包含中文原文和英文，代码中以英文为主，包含部分中文翻译

## Overall Summary(English translation)

If you are a beginner, you can run this project directly after downloading without having to understand the underlying architecture. You only need basic knowledge of Python and PyTorch, and reading the file descriptions. If you have any questions, you can contact us via email.

This README contains both the Chinese text and its English translation. The code primarily uses English, but also includes some Chinese translations.

This code implements the task of predicting the intermediate frame based on the preceding and subsequent frames, using an Encoder-Decoder architecture.


## 模型介绍

在Encoder部分，我使用了pvt_v2，即pyramid vision transformer。相比pvt_v1，pvt_v2主要在块编码时使用了overlapping编码，可以考虑到每个块之间的相关关系。不过根据pvt_v2原论文的实验部分的结论，它在attention部分相对于pvt_v1部分的改进几乎没有影响，而且通过阅读源码，我发现使用的是大小为7的平均池化，在不同大小的输入下泛化能力可能不佳，因此我使用了pvt_v1中原始的attention模块。

在Decoder部分，我们使用了反卷积和卷积相结合的解码方式。一共四次反卷积，每次包含一个反卷积操作和两个卷积操作。类似于Unet，本模型也考虑到了残差的影响，因此在解码时，每次反卷积后会和相同大小的Encoder结果在通道上进行叠加，能迫使模型更关注变化的部分，也避免模型过于模糊。

## Model Introduction(English translation)

In the Encoder part, I have used pvt_v2, which stands for Pyramid Vision Transformer. Compared to pvt_v1, pvt_v2 incorporates overlapping encoding during block encoding, considering the interrelationships between each block. However, according to the experimental findings in the original pvt_v2 paper, the improvements in the attention part compared to pvt_v1 are minimal. Additionally, upon examining the source code, I found that an average pooling operation of size 7 is used, which may not generalize well across different input sizes. Therefore, I opted to use the original attention module from pvt_v1.

In the Decoder part, we employ a combination of deconvolution and convolution operations. There are a total of four deconvolution steps, each consisting of one deconvolution operation and two convolution operations. Similar to Unet, this model also takes into account the impact of residuals. Hence, during decoding, after each deconvolution step, the result is element-wise added with the Encoder result of the same size in the channel dimension. This encourages the model to focus more on the changing parts while avoiding excessive blurriness.



*Download dataset: https://opendatalab.com/Vimeo90K/download*

> ## 文件说明
> 
> - train.py: 运行此文件可以进行训练（在此之前请先更改模型和测试图像的保存地址）
> 
> - data.py: 加载数据集
> 
> - model.py: 模型代码
> 
> - test.py: 调用预训练好的模型对现有视频进行预测
> 
> - test.mp4: 测试视频
> 
> - output_without_vfi.avi: 测试视频放慢四倍
> 
> - output.avi: 预测效果（与output_without_vfi.avi对比）
> 
> *由于Github文件大小限制(25MB)，我无法上传我预训练好的模型(约28MB)，如果你有兴趣，可以联系1793706453@qq.com*

> ## File Descriptions (English translation):
>
> - train.py: Run this file to perform training.
> 
> - data.py: Load the dataset.
> 
> - model.py: Model code.
> 
> - test.py: Call the pre-trained model to predict on existing videos.
> 
> - test.mp4: Test video.
> 
> - output_without_vfi.avi: Test video slowed down by a factor of four.
> 
> - output.avi: Predicted output (to be compared with output_without_vfi.avi).
> 
> *Due to the file size limit on GitHub (25MB), I am unable to upload my pre-trained model (approximately 28MB). If you are interested, please contact me at 1793706453@qq.com.*
