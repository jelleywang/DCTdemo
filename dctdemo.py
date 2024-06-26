# 參考來源：https://dev.to/marycheung021213/understanding-dct-and-quantization-in-jpeg-compression-1col

import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import skimage.io as io
from matplotlib.widgets import Slider, Button

# DCT matrix
N=8 # matrix shape: 8*8
DCT=np.zeros((N,N))
for m in range(N):
    for n in range(N):
        if m==0:
            DCT[m][n]=math.sqrt(1/N)
        else:
            DCT[m][n]=math.sqrt(2/N)*math.cos((2*n+1)*math.pi*m/(2*N))

# DCT basis image
basis=np.zeros((N*N,N*N))
for m in range(N):
    for n in range(N):
        pos_m=m*N
        pos_n=n*N
        DCT_v=DCT[m,:].reshape(-1,1)
        DCT_T_h=DCT.T[:,n].reshape(-1,N)
        basis[pos_m:pos_m+N,pos_n:pos_n+N]=np.matmul(DCT_v,DCT_T_h)

# Center values
basis+=np.absolute(np.amin(basis))
scale=np.around(1/np.amax(basis),decimals=3)
for m in range(basis.shape[0]):
    for n in range(basis.shape[1]):
        basis[m][n]=np.around(basis[m][n]*scale,decimals=3)

# Show basis image
# plt.figure(figsize=(4,4))
# plt.gray()
# plt.axis('off')
# plt.title('DCT Basis Image')
# plt.imshow(basis,vmin=0)

# DCT matrix
blocks=np.zeros((8*8,8))
for i in range(8):
    blocks[i*8][i]=1

# IDCT -> Original images
blocks_idct=np.zeros((8*8,8))
for i in range(8):
    block=blocks[i*8:i*8+8][:]
    data=cv2.idct(block)
    blocks_idct[i*8:i*8+8][:]=data

# Show DCT matrix
# plt.figure(figsize=(16,3))
# for i in range(8):
#     pos='18'+str(i+1)
#     pos=int(pos)
#     plt.subplot(pos)
#     block=blocks[i*8:i*8+8][:]
#     plt.gray()
#     plt.axis('off')
#     plt.imshow(block,vmin=0)

# Show original images
# plt.figure(figsize=(16,3))
# for i in range(8):
#     pos='18'+str(i+1)
#     pos=int(pos)
#     plt.subplot(pos)
#     block_idct=blocks_idct[i*8:i*8+8][:]
#     plt.gray()
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(block_idct,vmin=0)


# Read image
img=io.imread("lena.bmp",as_gray=True)
img = cv2.resize(img, (128, 128))

# Obtaining a mask through zigzag scanning
def z_scan_mask(C,N):
    mask=np.zeros((N,N))
    start=0
    mask_m=start
    mask_n=start
    for i in range(C):
        if i==0:
            mask[mask_m,mask_n]=1
        elif i<(N+1)*N/2:
            # If even, move upward to the right
            if (mask_m+mask_n)%2==0:
                mask_m-=1
                mask_n+=1
                # If it exceeds the upper boundary, move downward
                if mask_m<0:
                    mask_m+=1
                # If it exceeds the right boundary, move left
                if mask_n>=N:
                    mask_n-=1
            # If odd, move downward to the left
            else:
                mask_m+=1
                mask_n-=1
                # If it exceeds the lower boundary, move upward
                if mask_m>=N:
                    mask_m-=1
                # If it exceeds the left boundary, move right
                if mask_n<0:
                    mask_n+=1
            mask[mask_m,mask_n]=1
        elif i==(N+1)*N/2:
            # In the middle of the mask, turn
            mask_m+=1
            mask_n+=1
            if(mask_m>=N):
                mask_m-=1
            else:
                mask_n-=1
            mask[mask_m,mask_n]=1
        else:
            # If even, move upward to the right
            if (mask_m+mask_n)%2==0:
                mask_m-=1
                mask_n+=1
                if mask_n>=N:
                    mask_m+=2
                    mask_n-=1
            # If odd, move downward to the left
            else:
                mask_m+=1
                mask_n-=1
                # If it exceeds the lower boundary, move upward
                if mask_m>=N:
                    mask_m-=1
                    mask_n+=2
            mask[mask_m,mask_n]=1

    return mask

# overlaying the mask, discarding the high-frequency components
def Compress(img,mask,N):
    img_dct=np.zeros((img.shape[0]//N*N,img.shape[1]//N*N))
    for m in range(0,img_dct.shape[0],N):
        for n in range(0,img_dct.shape[1],N):
            block=img[m:m+N,n:n+N]
            # DCT
            coeff=cv2.dct(block)
            dct_block = coeff*mask
            # IDCT, but only the parts of the image where the mask has a value of 1 are retained
            iblock=cv2.idct(dct_block)
            img_dct[m:m+N,n:n+N]=iblock
    return img_dct

# Images keeping only 1, 3, and 10 low-frequency coefficients
# plt.figure(figsize=(16,4))
# plt.gray()
# plt.subplot(141)
# plt.title('Original image')
# plt.imshow(img)
# plt.axis('off')

# plt.subplot(142)
# plt.title('Keep 1 coefficient')
# plt.imshow(Compress(img,z_scan_mask(1,8),8))
# plt.axis('off')

# plt.subplot(143)
# plt.title('DCT Coefficients')
# plt.imshow(z_scan_mask(63,8))
# plt.axis('off')

# plt.subplot(144)
# plt.title('Keep 60 coefficients')
# plt.imshow(Compress(img,z_scan_mask(63,8),8))
# plt.axis('off')
# plt.show()


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////
# 初始化量化参数
quant_param = 1

# 创建图形和滑块
fig, ax = plt.subplots(1, 4, figsize=(15, 5))
plt.subplots_adjust(left=0.1, bottom=0.25)

# 显示原始图像
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original Image')

# 进行初始DCT并显示重建图像
compress_image = Compress(img,z_scan_mask(quant_param,8),8)
dct_coefficients_display = ax[1].imshow(z_scan_mask(quant_param,8), cmap='gray')
idct_img_display = ax[2].imshow(compress_image, cmap='gray')
residual_display = ax[3].imshow(img-compress_image, cmap='gray')
ax[1].set_title('DCT Coefficients')
ax[2].set_title('Reconstructed Image')
ax[3].set_title('Residual Image')

# 创建滑块
axcolor = 'lightgoldenrodyellow'
ax_quant = plt.axes([0.1, 0.1, 0.65, 0.05], facecolor=axcolor)
quant_slider = Slider(ax_quant, 'Quant Param', 1, 64, valinit=quant_param, valstep=1)

# 创建Apply按钮
apply_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
apply_button = Button(apply_ax, 'Apply', color=axcolor, hovercolor='0.975')

# 滑块更新函数
def update(val):
    global quant_param
    quant_param = int(quant_slider.val)

quant_slider.on_changed(update)

# 按钮点击事件
def apply(event):
    compress_image = Compress(img,z_scan_mask(quant_param,8),8)
    dct_coefficients_display.set_data(z_scan_mask(quant_param,8))
    idct_img_display.set_data(compress_image)
    residual_display.set_data(img-compress_image)
    fig.canvas.draw_idle()

apply_button.on_clicked(apply)

plt.show()