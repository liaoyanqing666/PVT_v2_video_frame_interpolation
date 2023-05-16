"""
This is the program of using the pre-trained model to predict any videos needed.
video_frame_interpolation1 and video_frame_interpolation2 are two callable interfaces with the following differences:

假设原始视频宽高为[VW, VH],模型接受的输入宽高为[W, H]

video_frame_interpolation1:
1. 对原始视频超采样至[VW*scale, VH*scale] (当scale!=1)
2. 将超采样后的视频的宽高padding至[W, H]的整数倍
3. 将padding后的视频切分为多个大小为[W, H]的块并分别预测
4. 将分别预测的结果合并并采样回[VW, VH]

video_frame_interpolation2:
1. 对原始视频采样至[W*scale[0], H*scale[1]]
2. 将超采样后的视频切分为多个大小为[W, H]的块并分别预测
3. 将分别预测的结果合并并采样回[VW, VH]

注意：如果运动物体正好处于拼接边界，可能会产生接缝明显或预测错误的问题。

English translation:
Assuming the original video has a width and height of [VW, VH], and the model accepts input with width and height [W, H].

video_frame_interpolation1:
1. Upsample the original video to [VW*scale, VH*scale] (when scale != 1).
2. Pad the upsampled video's width and height to the nearest multiple of [W, H].
3. Divide the padded video into multiple blocks of size [W, H] and predict each block separately.
4. Merge the individually predicted results and downsample back to [VW, VH].

video_frame_interpolation2:
1. Sample the original video to [W*scale[0], H*scale[1]].
2. Divide the sampled video into multiple blocks of size [W, H] and predict each block separately.
3. Merge the individually predicted results and sample back to [VW, VH].

Note: If moving objects are precisely located at the stitching boundary, it may result in noticeable seams or prediction errors.
"""

import shutil
import cv2
import os

import torch
from PIL import Image
from tqdm import tqdm
from model import *
import torchvision.transforms.functional as TF

# split the video
def video_split(video_dir, to_dir='split', encoding_method='png'):
    cap = cv2.VideoCapture(video_dir)
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    if len(os.listdir(to_dir)) != 0:
        shutil.rmtree(to_dir)
        os.mkdir(to_dir)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            file_name = os.path.join(to_dir, '{}.{}'.format(frame_count + 1, encoding_method))
            cv2.imwrite(file_name, frame)
            frame_count += 1
        else:
            break
    cap.release()


def super_sampling(width, height, dir='split'):
    img_files = os.listdir(dir)
    for img_file in tqdm(img_files):
        file_path = os.path.join(dir, img_file)
        img = Image.open(file_path)
        img_upscaled = img.resize((width, height), resample=Image.BICUBIC)
        img_upscaled.save(file_path)


def image_split_predict(video_width, video_height, model_dir='model_pth/model_iter53000.pth', H=256, W=448,
                        to_dir='predict', dir='split', subdir='temp', encoding_method='png'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VFImodel(H=H, W=W).to(device)
    state_dict = torch.load(model_dir, map_location=device)
    model.load_state_dict(state_dict)
    out_folder = os.path.join(dir, subdir)
    predict_folder = os.path.join(to_dir, subdir)
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(to_dir, exist_ok=True)
    if len(os.listdir(to_dir)) != 0:
        shutil.rmtree(to_dir)
        os.mkdir(to_dir)
    os.makedirs(predict_folder, exist_ok=True)

    img_files = os.listdir(dir)
    frames = len(img_files) - 1 # The extra 1 is the folder you just created
    frame_count = 0

    width, height = video_width, video_height
    block_size = (W, H)
    x_blocks = width // block_size[0]
    y_blocks = height // block_size[1]
    x_remainder = width % block_size[0]
    y_remainder = height % block_size[1]
    if x_remainder != 0:
        x_blocks += 1
    if y_remainder != 0:
        y_blocks += 1

    for i in tqdm(range(frames)):
        now = str(i + 1) + '.' + encoding_method
        frame_count += 1

        # split image 'now'
        img_path = os.path.join(dir, now)
        img = Image.open(img_path)
        for x in range(x_blocks):
            for y in range(y_blocks):
                left = x * block_size[0]
                upper = y * block_size[1]
                right = (x + 1) * block_size[0]
                lower = (y + 1) * block_size[1]
                # split and padding
                if x == x_blocks - 1 and x_remainder != 0:
                    right = left + x_remainder
                if y == y_blocks - 1 and y_remainder != 0:
                    lower = upper + y_remainder
                box = (left, upper, right, lower)
                block = img.crop(box)
                if right - left != block_size[0] or lower - upper != block_size[1]:
                    padded_block = Image.new('RGB', block_size, (255, 255, 255))
                    padded_block.paste(block, (0, 0))
                    block = padded_block
                out_path = os.path.join(out_folder, f'block_{x}_{y}_{frame_count}.{encoding_method}')
                block.save(out_path)

        # predict images
        if i > 0:
            for x in range(x_blocks):
                for y in range(y_blocks):
                    img1 = Image.open(os.path.join(out_folder, f'block_{x}_{y}_{frame_count-1}.{encoding_method}'))
                    img2 = Image.open(os.path.join(out_folder, f'block_{x}_{y}_{frame_count}.{encoding_method}'))
                    img1_tensor = TF.to_tensor(img1).unsqueeze(0).to(device)
                    img2_tensor = TF.to_tensor(img2).unsqueeze(0).to(device)
                    with torch.no_grad():
                        predict = model(img1_tensor, img2_tensor).cpu()
                    predict_img = TF.to_pil_image(predict.squeeze())
                    predict_img.save(os.path.join(predict_folder, f'block_{x}_{y}_{frame_count-1}.{encoding_method}'))

            # Composite predicted image
            recomposed_img = Image.new('RGB', (width, height), (255, 255, 255))
            for x in range(x_blocks):
                for y in range(y_blocks):
                    left = x * block_size[0]
                    upper = y * block_size[1]
                    block_path = os.path.join(predict_folder, f'block_{x}_{y}_{frame_count-1}.{encoding_method}')
                    block = Image.open(block_path)
                    recomposed_img.paste(block, (left, upper))
            recomposed_img = recomposed_img.crop((0, 0, video_width, video_height))
            recomposed_img.save(os.path.join(to_dir, f'{frame_count-1}.{encoding_method}'))


def compose_video(frames, frame_rate, frame_width, frame_height, split_dir='split', predict_dir='predict', encoding_method='png', output_path='output.avi'):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))
    for i in range(frames):
        img1_path = os.path.join(split_dir, str(i + 1) + '.' +encoding_method)
        img1 = cv2.imread(img1_path)
        img1 = cv2.resize(img1, (frame_width, frame_height))
        out.write(img1)
        if i < frames - 1:
            img2_path = os.path.join(predict_dir, str(i + 1) + '.' +encoding_method)
            img2 = cv2.imread(img2_path)
            img2 = cv2.resize(img2, (frame_width, frame_height))
            out.write(img2)


def video_frame_interpolation1(video, output_name='output.avi', H=256, W=448, split_dir='split', predict_dir='predict', subfolder_name='temp',
                              scale=1, encoding_method='png', slow_down=False, model='model_pth/model_iter53000.pth'):
    """
    Introduction of this function is at the top.

    :param video: Input video of any size.
    :param output_name: Name of the output video.
    :param H: H of the pretrained model.
    :param W: W of the pretrained model.
    :param split_dir: The folder name of split frames of the input video.
    :param predict_dir: The folder name of predicted frames of the video.
    :param subfolder_name: The name of subfolder to store the split images.
    :param scale: The magnification factor of the original frames. (Please refer to the program introduction at the top for specific operations.)
    :param encoding_method: The coded format of the frames.
    :param slow_down: Whether the output video a slow-motion version of the input video or an increase in frame rate.
    :param model: The direction of the pretrained model.
    """
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print('Begin video splitting')
    video_split(video, split_dir, encoding_method)
    print('Complete video splitting\n')
    if scale > 1:
        print('Begin super sampling')
        super_sampling(frame_width * scale, frame_height * scale, split_dir)
        print('Complete super sampling\n')
    print('Begin predicting')
    image_split_predict(
        video_width=frame_width * scale,
        video_height=frame_height * scale,
        model_dir=model,
        H=H, W=W,
        to_dir=predict_dir,
        dir=split_dir,
        subdir=subfolder_name,
        encoding_method=encoding_method
    )
    print('Complete predicting\n')
    if not slow_down:
        frame_rate *= 2
    print('Begin composing')
    compose_video(
        frames=frame,
        frame_rate=frame_rate,
        frame_width=frame_width,
        frame_height=frame_height,
        split_dir=split_dir,
        predict_dir=predict_dir,
        encoding_method=encoding_method,
        output_path=output_name
    )
    print('Complete composing\n')
    print('Complete all\n\n')


def video_frame_interpolation2(video, output_name='output.avi', H=256, W=448, scale=(1, 1), split_dir='split',
                               predict_dir='predict', subfolder_name='temp',
                               encoding_method='png', slow_down=False, model='model_pth/model_iter53000.pth'):
    """
    Introduction of this function is at the top.

    :param video: Input video of any size.
    :param output_name: Name of the output video.
    :param H: H of the pretrained model.
    :param W: W of the pretrained model.
    :param split_dir: The folder name of split frames of the input video.
    :param predict_dir: The folder name of predicted frames of the video.
    :param subfolder_name: The name of subfolder to store the split images.
    :param scale: The magnification factor of the original frames. (Please refer to the program introduction at the top for specific operations.)
    :param encoding_method: The coded format of the frames.
    :param slow_down: Whether the output video a slow-motion version of the input video or an increase in frame rate.
    :param model: The direction of the pretrained model.
    """
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print('Begin video splitting')
    video_split(video, split_dir, encoding_method)
    print('Complete video splitting\n')
    W_scale, H_scale = scale
    if scale != (1, 1):
        print('Begin super sampling')
        super_sampling(width=W * W_scale, height=H * H_scale, dir=split_dir)
        print('Complete super sampling\n')
    print('Begin predicting')
    image_split_predict(
        video_width=W * W_scale,
        video_height=H * H_scale,
        model_dir=model,
        H=H, W=W,
        to_dir=predict_dir,
        dir=split_dir,
        subdir=subfolder_name,
        encoding_method=encoding_method
    )
    print('Complete predicting\n')
    if not slow_down:
        frame_rate *= 2
    compose_video(
        frames=frame,
        frame_rate=frame_rate,
        frame_width=frame_width,
        frame_height=frame_height,
        split_dir=split_dir,
        predict_dir=predict_dir,
        encoding_method=encoding_method,
        output_path=output_name
    )
    print('Complete composing\n')
    print('Complete all\n\n')


if __name__ == '__main__':
    # examples
    video_frame_interpolation1(video='test.mp4', scale=1, output_name='output.avi', slow_down=True)
    video_frame_interpolation1(video='output.avi', scale=1, output_name='output.avi', slow_down=True)
    # video_frame_interpolation2(video='test.mp4',scale=(2, 2),output_name='output.avi',slow_down=True)
    # video_frame_interpolation2(video='output.avi',scale=(2, 2),output_name='output.avi',slow_down=True)