"""
uJPEG
Yunho Cho
"""

from pathlib import Path

import numpy as np
import typer
from PIL import Image

def rgb_to_ycbcr(image_rgb):
  coefs = np.array([
    [ 0.299,     0.587,     0.114   ],
    [-0.168736, -0.331264,  0.5     ],
    [ 0.5,      -0.418688, -0.081312]
  ])
  offset = np.array([0, 128, 128])

  image_ycbcr = image_rgb @ coefs.T + offset

  # or more readable (but slightly slower) way:
  # R = image_rgb[:, :, 0]
  # G = image_rgb[:, :, 1]
  # B = image_rgb[:, :, 2]

  # Y  =  0.299 * R + 0.587 * G + 0.114 * B
  # Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
  # Cr =  0.5 * R - 0.418688 * G - 0.081312 * B + 128
  # iamge_ycbcr = np.stack([Y, Cb, Cr], axis=-1)
  
  return np.clip(image_ycbcr, 0, 255).astype(np.uint8)

def pad_image_to_block_size(image, block_size, mode="edge"):
  height, width = image.shape[:2]
  vert_rem, horz_rem = height % block_size, width % block_size
  vert_pad = block_size - vert_rem if vert_rem != 0 else 0
  horz_pad = block_size - horz_rem if horz_rem != 0 else 0
  pads = [
    (0, vert_pad),
    (0, horz_pad),
  ]
  if image.ndim == 3: 
    pads += [(0, 0)]
  return np.pad(image, pads, mode=mode)

def subsample_420(channel):
  height, width = channel.shape
  blocks = channel.reshape(height // 2, 2, width // 2, 2)
  subsampled = blocks.mean(axis=(1, 3))
  return subsampled.astype(np.uint8)

def main(
  path: str,
  show: bool = False
):
  original_image = Image.open(path)
  image_rgb = np.array(original_image.convert("RGB"))

  # Stage 1. RGB -> YCbCr color conversion
  image_ycbcr = rgb_to_ycbcr(image_rgb)
  image_ycbcr_p = pad_image_to_block_size(image_ycbcr, block_size=2)
  Y  = image_ycbcr_p[:, :, 0]
  Cb = image_ycbcr_p[:, :, 1]
  Cr = image_ycbcr_p[:, :, 2]

  # Stage 2. Chroma subsampling
  Cb_sub = subsample_420(Cb)
  Cr_sub = subsample_420(Cr)

  if show:
    # original_image.show()
    Image.fromarray(image_ycbcr_p).show()

if __name__ == "__main__":
  typer.run(main)
