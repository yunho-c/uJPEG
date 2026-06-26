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

def generate_dct_matrix(N=8):
  D = np.zeros((N, N))
  for i in range(N):
    alpha = np.sqrt(1 / N) if i == 0 else np.sqrt(2 / N)
    for j in range(N):
      D[i, j] = alpha * np.cos((np.pi * (2 * j + 1) * i) / (2 * N))
  return D

D = generate_dct_matrix()

def apply_dct(channel, inverse=False):
  height, width = channel.shape
  # reshape into 8x8 blocks using "Grid of Grids" trick
  blocks = channel.reshape(height // 8, 8, width // 8, 8)

  # perform matrix multiplication on every block at once
  if not inverse: # forward
    dct_blocks = D @ blocks @ D.T
  else: # inverse
    dct_blocks = D.T @ blocks @ D

  # reshape back into 2D image
  return dct_blocks.transpose(0, 2, 1, 3).reshape(height, width)

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

  # Level shifting
  Y      -= 128
  Cb_sub -= 128
  Cr_sub -= 128

  # Stage 3. Discrete cosine transform (DCT)
  Y      = pad_image_to_block_size(Y     , block_size=8)
  Cb_sub = pad_image_to_block_size(Cb_sub, block_size=8)
  Cr_sub = pad_image_to_block_size(Cr_sub, block_size=8)

  Y_dct      = apply_dct(Y)
  Cb_sub_dct = apply_dct(Cb_sub)
  Cr_sub_dct = apply_dct(Cr_sub)
  
  if show:
    # original_image.show()
    Image.fromarray(image_ycbcr_p).show()

if __name__ == "__main__":
  typer.run(main)
