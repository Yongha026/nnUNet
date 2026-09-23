import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# 1. .npy 파일 로드
# mask = np.load('./test_gbc/000000341215.npy')
mask = np.load('./mask_gt.npy')
mask_resized = cv.resize(mask, (400,640),interpolation=cv.INTER_NEAREST)

# 2. 이미지 띄우기
plt.figure(figsize=(5,8))
plt.imshow(mask_resized, cmap='viridis')  # 이진 마스크는 'gray', 멀티클래스는 'viridis' 등 추천
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.tight_layout()

plt.savefig('mask.png')
