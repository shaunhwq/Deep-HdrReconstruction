import argparse
import numpy as np
import os
import torch

from torchvision import transforms

import cv2
from lib.image import unnormalize
from lib.io import load_ckpt
from lib.util import get_saturated_regions
from network.softconvmask import SoftConvNotLearnedMaskUNet
from tqdm import tqdm


# https://stackoverflow.com/questions/50748084/convert-exr-to-jpeg-using-imageio-and-python
# Gamma usually use 2.2
# Gamma adjustment should be done before mapping to uint8

def pyr_crop(img: np.array, num_layers=3) -> np.array:
    """
    Prevents errors for pyramid style networks by center cropping such that dimensions are divisible by num_layers power of 2.

    :param img: input image to be cropped
    :param num_layers: Crop input image to be able to fit a network with pyramid with num_layers layers.
    :returns: cropped image.
    """
    h, w, _ = img.shape
    des_h, des_w = np.floor(np.array([h, w]) / pow(2, num_layers - 1)).astype(np.int32) * pow(2, num_layers - 1)
    w_start, h_start = (w - des_w) // 2, (h - des_h) // 2
    return img.copy()[h_start: h_start + des_h, w_start: w_start + des_w, ::]


def restore_pyr_crop(img: np.array, original_shape: tuple) -> np.array:
    """
    Zero pads an image that has been cropped to its original size

    :param img: Cropped image (using pyr_crop for example)
    :param original_shape: Original shape of the image before it was cropped in the format (h, w, c)
    :returns: Zero padded img with size original_shape
    """
    oh, ow, _ = original_shape
    ih, iw, _ = img.shape
    if oh == ih and ow == iw:
        return img

    # Zero pad and place cropped image in center
    restored = np.zeros(original_shape, dtype=np.float32)
    start_h, start_w = int((oh - ih) // 2), int((ow - iw) // 2)
    restored[start_h: start_h + ih, start_w: start_w + iw, ::] = img

    return restored


def get_model_input(img: np.array, device: str, tv_transforms):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.

    # get saturation mask
    conv_mask = 1 - get_saturated_regions(img)
    conv_mask = torch.from_numpy(conv_mask).permute(2, 0, 1)

    img = tv_transforms(img)
    img, conv_mask = torch.stack([img]), torch.stack([conv_mask])

    img = img.to(device)
    conv_mask = conv_mask.to(device)
    return img, conv_mask


def visualize_model_output(img, mask, model_out):
    img, mask = img.cpu(), mask.cpu()
    img = unnormalize(img).permute(0, 2, 3, 1).numpy()[0, :, :, :]
    mask = mask.permute(0, 2, 3, 1).numpy()[0, :, :, :]
    model_out = model_out.cpu().permute(0, 2, 3, 1).numpy()[0, :, :, :]

    y_predict = np.exp(model_out) - 1
    gamma = np.power(img, 2)
    H = mask * gamma + (1 - mask) * y_predict

    H = cv2.cvtColor(H, cv2.COLOR_RGB2BGR)
    gamma = cv2.cvtColor(gamma, cv2.COLOR_RGB2BGR)

    return H, gamma


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single Image HDR Reconstruction Using a CNN with Masked Features and Perceptual Loss")
    parser.add_argument('--input_dir', '-t', type=str, required=True, help='Input images directory.')
    parser.add_argument('--output_dir', '-o', type=str, required=True, help='Path to output directory.')
    parser.add_argument('--weights_path', '-w', type=str, help='Path to the trained CNN weights.', default="checkpoints/ldr2hdr.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    args.train = False

    if not os.path.exists(args.output_dir):
        os.makedirs(os.path.join(args.output_dir, "DeepHDR_gamma"))
        os.makedirs(os.path.join(args.output_dir, "DeepHDR_h"))

    assert os.path.exists(args.weights_path), "Unable to find pretrained weights"
    model = SoftConvNotLearnedMaskUNet().to(args.device)
    model.load_state_dict(torch.load(args.weights_path, map_location=args.device)["model"])
    model.eval()

    # load test data.
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_paths = [os.path.join(args.input_dir, img_path) for img_path in os.listdir(args.input_dir) if img_path[0] != "."]

    with torch.no_grad():
        for img_path in tqdm(image_paths, total=len(image_paths), desc="Running DeepHDR..."):
            # Load and transform image
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            #in_img, in_mask = get_model_input(image, args.device, img_transform)
            cropped_img = pyr_crop(image, num_layers=8)
            in_img, in_mask = get_model_input(cropped_img, args.device, img_transform)

            # Model inference
            model_output = model(in_img, in_mask)

            # Transform model output and save
            out_h, out_gamma = visualize_model_output(in_img, in_mask, model_output)

            padded_h = restore_pyr_crop(out_h, image.shape)
            padded_gamma = restore_pyr_crop(out_gamma, image.shape)

            new_name = os.path.splitext(os.path.basename(img_path))[0] + ".hdr"
            output_path = os.path.join(args.output_dir, "DeepHDR_h", new_name)
            cv2.imwrite(output_path, padded_h)

            new_name = os.path.splitext(os.path.basename(img_path))[0] + ".hdr"
            output_path = os.path.join(args.output_dir, "DeepHDR_gamma", new_name)
            cv2.imwrite(output_path, padded_gamma)
