

import argparse
import os

import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm

CLIP_LOWER_PERCENTILE = 0.5
CLIP_UPPER_PERCENTILE = 99.5


def convert_dicom_to_png16(dicom_path: str, png_path: str) -> None:
    ds = pydicom.dcmread(dicom_path)
    pixels = ds.pixel_array.astype(np.float64)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept

    lo, hi = np.percentile(pixels, [CLIP_LOWER_PERCENTILE, CLIP_UPPER_PERCENTILE])
    pixels = np.clip(pixels, lo, hi)

    if hi > lo:
        normalized = (pixels - lo) / (hi - lo)
    else:
        normalized = np.zeros_like(pixels)
    png16 = (normalized * 65535.0).round().astype(np.uint16)
    Image.fromarray(png16, mode="I;16").save(png_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Root directory with DICOM files")
    parser.add_argument("--output", required=True, help="Output directory for 16-bit PNGs")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    dicom_files = []
    for dirpath, _, filenames in os.walk(args.input):
        for name in filenames:
            if name.lower().endswith((".dcm", ".dicom")):
                dicom_files.append(os.path.join(dirpath, name))

    for dcm_path in tqdm(sorted(dicom_files), desc="Converting DICOM to PNG"):
        rel = os.path.relpath(dcm_path, args.input)
        out_path = os.path.join(args.output, os.path.splitext(rel)[0] + ".png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        convert_dicom_to_png16(dcm_path, out_path)

    print(f"Converted {len(dicom_files)} DICOM files to {args.output}")


if __name__ == "__main__":
    main()
