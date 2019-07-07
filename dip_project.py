from model import *
from data import *
import os
from matplotlib import pyplot
import cv2 as cv
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
DATAPATH='D:\Semester6\dataset\hrainx'
DATA_testx='D:\Semester6\dataset\hestx'
DATA_testy='D:\Semester6\dataset\hesty'
myskin = trainGenerator(2,DATAPATH,os.path.join(DATAPATH,'image'),os.path.join(DATAPATH,'mask'),data_gen_args,save_to_dir ='D:\Semester6\dataset\output')

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
# model.fit_generator(myskin,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])


DATA_testx='D:\Semester6\dataset\hestx'

testskin = testGenerator(DATAPATH)
# results1=model.predict_generator(myskin,20,verbose=1)
results = model.predict_generator(testskin,40,verbose=1)
# plt.imshow
saveResult("D:\Semester6\dataset\hestx",results)
# saveResult("D:\Semester6\dataset\hrainy",results1 )
# hist = model.fit(myskin, epochs=300, batch_size=16, validation_data=(testskin), verbose=1)
model.save("model.h5")
accuracy = model.evaluate(testskin,results,16)
print("Accuracy: ", accuracy[1])