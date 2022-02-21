import numpy as np
from skimage import measure


def calculate_sad(img1, img2):
    if img1.shape != img2.shape:
        print("Images don't have the same shape.")
        return

    score = np.sum(abs(np.array(img1, dtype=np.float32) -
                       np.array(img2, dtype=np.float32)))

    return score


def calculate_mae(img1, img2):
    if img1.shape != img2.shape:
        print("Images don't have the same shape.")
        return

    score = np.sum(abs(np.array(img1, dtype=np.float32) -
                       np.array(img2, dtype=np.float32)))
    score = score / (img1.shape[0] * img1.shape[1])
    return score


def calculate_ssd(img1, img2):
    """Computing the sum of squared differences (SSD) between two images."""
    if img1.shape != img2.shape:
        print("Images don't have the same shape.")
        return
    score = np.sum((np.array(img1, dtype=np.float32) -
                    np.array(img2, dtype=np.float32))**2)
    return score


def calculate_rmse(img1, img2):
    """Computing the sum of squared differences (SSD) between two images."""
    if img1.shape != img2.shape:
        print("Images don't have the same shape.")
        return
    score = np.sum((np.array(img1, dtype=np.float32) -
                    np.array(img2, dtype=np.float32))**2)
    score = score / (img1.shape[0] * img1.shape[1])
    score = np.sqrt(score)
    return score


# PSNRは高いほど画質が良い
# 20db以下だと劣化が目立つ
def calculate_psnr(img1, img2, data_range=1):
    if img1.shape != img2.shape:
        print("Images don't have the same shape.")
        return
    # print(img1)
    np.seterr(divide='ignore')  # 0除算のRuntimeWarningのみを無視扱いとする
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    score = 10 * np.log10((data_range ** 2) / mse)
    return score

# SSIMは1-~1で表され，1に近いほど似ている画像とされる


def calculate_ssim(img1, img2):
    if img1.shape != img2.shape:
        print("Images don't have the same shape.")
        return
    score = measure.compare_ssim(img1, img2, multichannel=True)
    return score
