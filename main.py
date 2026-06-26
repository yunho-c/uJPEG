"""
uJPEG
Yunho Cho
"""

from pathlib import Path

import numpy as np
import typer
from PIL import Image

def main(
  path: str, 
  show: bool = False
):
  original_image = Image.open(path)

  if show:
    original_image.show()

if __name__ == "__main__":
  typer.run(main)
