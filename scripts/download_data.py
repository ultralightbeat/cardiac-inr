import os
import zipfile
import requests
from tqdm import tqdm


def download_and_extract_dataset():
    # 目标存放目录（根据你的 pipeline.py，代码默认从 data/acdc/ 读取数据）
    target_dir = os.path.join("data", "acdc")
    os.makedirs(target_dir, exist_ok=True)

    zip_path = os.path.join(target_dir, "acdc_dataset.zip")

    # Kaggle 直链
    url = "https://storage.googleapis.com/kaggle-data-sets/4020939/7040769/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20261102%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20261102T045423Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=70fbe2f314f8553a75c956af45c049e35dee93b6622779e89989e1e9506ed55d925a467bb7907edd63c0eef94e244e9e397711954057b4a88d5cfd1fde9254477243898102f17d912e1bd67d6cf0aa5aaa71d61eca27d666b329bf2dcc0538eb9b917d80cddf6c55aa2277fd324c4206ce2f329aa722cdacbb2c94bfbf7e48c8d2d68b3689d6691e4089cb9645b0813e35797007724fcac7e4275169a645116379ea1b73c1784e15f01fc691beaee58bf4a05187fc58343b3d4627d887a0fee14d319d0b6677e1f479ab205062a9a1c02125536fb65c5da1296e2d4f54aafa10580e701e4438eeed8e2d2ba23f0af47a100155215b01dce476c203c77c86a825"

    # 检查是否已经存在解压后的患者文件夹（例如 patient051），如果存在则跳过
    if os.path.exists(os.path.join(target_dir, "patient051")):
        print("Patient data already exists; skipping download and extraction.")
        return

    # 1. 下载压缩包并显示进度条
    print("Downloading the ACDC dataset...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(zip_path, "wb") as f, tqdm(
            desc="Download progress", total=total_size, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            f.write(data)
            bar.update(len(data))

    # 2. 解压数据集
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    # 3. 清理压缩包
    os.remove(zip_path)
    print("The dataset has been downloaded and decompressed successfully, and it is now stored in the 'data/acdc/' directory!")


if __name__ == "__main__":
    download_and_extract_dataset()