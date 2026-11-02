import os
import shutil
import zipfile
import requests
from tqdm import tqdm


def download_and_extract_dataset():
    # 1. 自动定位项目根目录（从 scripts/ 目录返回上一级）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    # 2. 确保目标目录位于项目根目录下的 data/acdc
    target_dir = os.path.join(project_root, "data", "acdc")
    os.makedirs(target_dir, exist_ok=True)
    print(f"Target directory confirmed: {target_dir}")

    # 检查是否已有患者文件夹（如 patient051 已存在则跳过）
    if any(d.startswith("patient") for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))):
        print("Detected existing ACDC patient data, skipping download.")
        return

    zip_path = os.path.join(target_dir, "acdc_dataset.zip")
    url = "https://storage.googleapis.com/kaggle-data-sets/4020939/7040769/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20261102%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20261102T045423Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=70fbe2f314f8553a75c956af45c049e35dee93b6622779e89989e1e9506ed55d925a467bb7907edd63c0eef94e244e9e397711954057b4a88d5cfd1fde9254477243898102f17d912e1bd67d6cf0aa5aaa71d61eca27d666b329bf2dcc0538eb9b917d80cddf6c55aa2277fd324c4206ce2f329aa722cdacbb2c94bfbf7e48c8d2d68b3689d6691e4089cb9645b0813e35797007724fcac7e4275169a645116379ea1b73c1784e15f01fc691beaee58bf4a05187fc58343b3d4627d887a0fee14d319d0b6677e1f479ab205062a9a1c02125536fb65c5da1296e2d4f54aafa10580e701e4438eeed8e2d2ba23f0af47a100155215b01dce476c203c77c86a825"

    print("Attempting to automatically download the dataset...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(zip_path, "wb") as f, tqdm(
                desc="Download Progress", total=total_size, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                bar.update(len(data))
    except Exception as e:
        print(f"\n❌ Automatic download failed, error: {e}")
        print("💡 Tip: The download link may have expired. Please manually place the patient folders into data/acdc/!")
        return

    # 解压到临时文件夹
    extract_temp_dir = os.path.join(target_dir, "temp_extracted")
    os.makedirs(extract_temp_dir, exist_ok=True)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_temp_dir)

    print("Organizing directory structure...")
    # 自动在解压出的多层目录（如 database/training/）中搜寻 patient 文件夹，并提权移动到 data/acdc/ 下
    for root, dirs, files in os.walk(extract_temp_dir):
        for d in dirs:
            if d.startswith("patient"):
                src_path = os.path.join(root, d)
                dst_path = os.path.join(target_dir, d)
                if not os.path.exists(dst_path):
                    shutil.move(src_path, dst_path)

    # 清理临时压缩包和缓存文件夹
    shutil.rmtree(extract_temp_dir, ignore_errors=True)
    if os.path.exists(zip_path):
        os.remove(zip_path)

    print("\n✅ Dataset is ready!")


if __name__ == "__main__":
    download_and_extract_dataset()
