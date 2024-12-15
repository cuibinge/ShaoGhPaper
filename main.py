from mynet import *
from model import unet
from data import *
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from utils import plotHistory, plotFeaturemap
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from sklearn.model_selection import train_test_split
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)



KTF.set_session(sess)


savePath = 'logs/pse_net3-2019-05-15_21-48/'
pretrained_weights = savePath + 'unet-v1-improvement-041-0.9615.hdf5'
image = np.load('./imagesDataset_.npy')
label = np.load('./labelsDataset_.npy')
land = np.load('./imagesDataset_.npy')

train_image,val_image,train_label,val_label = train_test_split(image,label,test_size=0.05, random_state=0)

# train_image = image[0:6300,:,:,:]
# train_label = label[0:6300,:,:,:]
# val_image = image[6300:,:,:,:]
# val_label = label[6300:,:,:,:]

train_land = land[0:450,:,:,:]
train_land_zero = np.zeros((450,256,256,1))
val_land = land[450:,:,:,:]
val_land_zero = np.zeros((50,256,256,1))

train_image = np.vstack((train_image, train_land))
# 调整 train_label 的形状
# 调整 train_land_zero 的形状
train_land_zero = train_land_zero[:, :128, :128, :]  # 截取前 128x128 的区域
train_label = np.vstack((train_label, train_land_zero))
val_image = np.vstack((val_image, val_land))
# 调整 val_land_zero 的形状
# 调整 val_label 的形状
# 调整 val_land_zero 的形状
val_land_zero = val_land_zero[:, :128, :128, :]  # 截取前 128x128 的区域
val_label = np.vstack((val_label, val_land_zero))
print("pre train_image",train_image.shape)  # 打印 train_image 的形状
# --------------
from skimage.transform import resize

# 假设 train_image 的形状是 (1027, 128, 128, 3)
train_image = resize(train_image, (1027, 256, 256, 3))  # 调整为 (1027, 256, 256, 3)
print("post train_image",train_image.shape)  # 打印 train_image 的形状

print("pre train_label",train_label.shape)  # 打印 train_label 的形状

# 假设 train_label 的形状是 (1027, 128, 128, 1)
train_label = resize(train_label, (1027, 256, 256, 1))  # 调整为 (1027, 256, 256, 1)

print("post train_label",train_label.shape)  # 打印 train_label 的形状

print("pre val_image",val_image.shape)
val_image = resize(val_image, (81, 256, 256, 3))  # 调整为 (189, 256, 256, 3)
print("post val_image",val_image.shape)

print("pre val_image",val_label.shape)
val_label = resize(val_label, (81, 256, 256, 1))  # 调整为 (189, 256, 256, 3)
print("post val_image",val_label.shape)
# ----------------


# train_black_label = np.zeros((4005,128,128,1))
# black_val = black[2200:2800,:,:,:]
# val_black_label = np.zeros((600,128,128,1))

IMAGE_SIZE = 256
LR = 0.0001
epochs = 100
batch_size = 2
# =============================================================================
# 模型
# train_image_zero = np.zeros((500,128,128,3))
# train_label_zero = np.zeros((500,128,128,1))
# val_image_zero = np.zeros((50,128,128,3))
# val_label_zero = np.zeros((50,128,128,1))
model = pse_net3(2, (IMAGE_SIZE,IMAGE_SIZE,3), epochs, batch_size, LR,
    Falg_summary=True, Falg_plot_model=False, pretrained_weights = False)

# =============================================================================

savePath = mkSaveDir('zhaounet')
# 使用保存点
checkpointPath= savePath + "/unet-improvement-{epoch:03d}-{val_acc:.4f}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(checkpointPath, monitor='val_loss', verbose=1,
                             save_best_only=True, save_weights_only=True, mode='auto', period=1)
EarlyStopping = EarlyStopping(monitor='val_acc', patience=50, verbose=1)
tensorboard = TensorBoard(log_dir=savePath, histogram_freq=0)
callback_lists = [tensorboard, EarlyStopping, checkpoint]
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
# =============================================================================
# True: 读取图像;False: Train with npy file
readImg = False
if readImg:
    myGene = trainGenerator(2,'dataset/train','images','groundTruth',save_to_dir = None)
    model.fit_generator(myGene,steps_per_epoch=30,epochs=epochs,callbacks=callback_lists)
else:
    # train_image, train_GT, valid_image, valid_GT = readNpy()
    # train_image = np.vstack((train_image, black))
    # train_GT = np.vstack((train_GT, train_black_label))
    # valid_image = np.vstack((valid_image, black_val))
    # valid_GT = np.vstack((valid_GT, val_black_label))

    History = model.fit(train_image, train_label, batch_size=batch_size, validation_data=(val_image, val_label),
        epochs=epochs, verbose=1, shuffle=True, class_weight='auto', callbacks=callback_lists)
    with open(savePath + '/log_128.txt','w') as f:
        f.write(str(History.history))
#model.save_weights(savePath + '/save_weights.h5')

# 绘制accurate和loss曲线
plotHistory(History, savePath)
# 绘制FeatureMap
#plotFeaturemap(valid_image[0:1], model, savePath)
# =============================================================================
# 预测
# f_names = glob.glob(savePath + '/*.hdf5')
# for i, f_name in enumerate(f_names):
    # modelTest(f_name, IMAGE_SIZE = 1024)

# =============================================================================