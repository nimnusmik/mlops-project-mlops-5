import os
import random
import hashlib
import sys # logger를 위해 추가

import numpy as np


def init_seed():
    np.random.seed(0)
    random.seed(0)


def project_path():
    return os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "..",
        ".."
    )


def model_dir(model_name):
    return os.path.join(
        project_path(),
        "models",
        model_name
    )


def auto_increment_run_suffix(name: str, pad=3):  
    suffix = name.split("-")[-1]
    next_suffix = str(int(suffix) + 1).zfill(pad)
    return name.replace(suffix, next_suffix)


def calculate_hash(filename):  
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def save_hash(dst, logger):  
    hash_ = calculate_hash(dst)
    dst_base, _ = os.path.splitext(dst)  
    hash_file_path = f"{dst_base}.sha256"  
    try:
        with open(hash_file_path, "w") as f:
            f.write(hash_)
        logger.write(f"해시 저장 완료: {dst} -> {hash_file_path}")  
    except Exception as e:
        logger.write(f"[ERROR] 해시 저장 실패: {dst} (오류: {e})", print_error=True)  


def read_hash(dst, logger):  
    dst_base, _ = os.path.splitext(dst)  
    hash_file_path = f"{dst_base}.sha256" 
    if os.path.exists(hash_file_path):
        try:
            with open(hash_file_path, "r") as f:
                hash_val = f.read().strip()
            logger.write(f"해시 파일 읽기 완료: {hash_file_path}")  
            return hash_val
        except Exception as e:
            logger.write(f"[ERROR] 해시 파일 읽기 실패: {hash_file_path} (오류: {e})", print_error=True)  
            return None
    else:
        logger.write(f"[WARN] 해시 파일이 존재하지 않습니다: {hash_file_path}", print_also=True)  
        return None
