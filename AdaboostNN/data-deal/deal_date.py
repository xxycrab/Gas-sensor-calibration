import os
import sys
import numpy
from PIL import Image
import os.path
import glob
import cPickle 
import os.path 
import shutil 
import time,  datetime
#import config_parser
"""
    THIS CLASS WILL DEAL THE IMAGE TO PICKLE FILE, THE DTYPE: INT IS INT64
                                                   THE DTYPE: FLOAT IS FLOAT32
"""
class DealDate(object):
    def __init__(self,image_path="../data/image",size_x=28,size_y=28,channel=3 , \
                 valid_rate = 0.1, test_rate = 0.1):
        self.image_path = image_path
        self.size_x     = size_x
        self.size_y     = size_y
        self.channel    = channel
        self.valid_rate = valid_rate
        self.test_rate  = test_rate
        self.datasets_n    = {}

    # GET THE DIRCTORY FORM THE IMAGE_PATH
    def get_dirc_list(self):
        try:
            tems = os.listdir(self.image_path)
            dirs = []
            for f in tems:
                if os.path.isdir(os.path.join(self.image_path,f)):
                    dirs.append(f)
        except Exception as err:
            print err
        return dirs

    # GET THE FILE FORM THE IMAGE_PATH
    def get_file_list(self):
        try:
            tems = os.listdir(self.image_path)
            files = []
            for f in tems:
                if os.path.isfile(os.path.join(self.image_path,f)):
                    files.append(f)
        except Exception as err:
            print err
        return files

    # REMOVE THE PICKLE FILE FORM THE IMAGE_PATH
    def remove_pkl_list(self):
        try:
            tems = os.listdir(self.image_path)
            files = []
            for f in tems:
                if os.path.isfile(os.path.join(self.image_path,f)) and \
                    f[-4:] == ".pkl":
                    os.remove(os.path.join(self.image_path,f))
        except Exception as err:
            print err
        return files

    # RESIZE THE IMAGE TO THE FIXED SIZE. 
    def convert_img(self,img_list ,source_img_path, dest_img_path, width=48, height=36):
        if not os.path.isdir(dest_img_path):
            os.makedirs(dest_img_path)
        for img in img_list:
            try:
                img_path = os.path.join(source_img_path,img)
                img_obj  = Image.open(img_path) 
                new_img=img_obj.resize((width,height),Image.BILINEAR) 
                out_path =  os.path.join(dest_img_path,img+'.jpg')
                new_img.save(out_path)
            except Exception as err:
                print err

    # CONVERT THE IMAGE TO ARRAY, THEN SAVE IT AS THE PICKLE FILE.
    def convert_pic(self):

        dirs = self.get_dirc_list() 
        for d in dirs:
            image_array = []
            dir_path = os.path.join(self.image_path,str(d))
            images = os.listdir(dir_path)
            m = 0
            for image in images:
                try:
                    img_path = os.path.join(dir_path,image)
                    fp = open(img_path, 'r')
                    tem = numpy.asarray( Image.open(fp).resize((self.size_x,self.size_y),       \
                                         Image.BILINEAR),dtype='float32')/255.0
                    image_array.append(tem.reshape((1,self.size_x * self.size_y * self.channel)))
                    m = m + 1
                    fp.close() 
                except Exception as err:
                    print err
                    continue
            self.datasets_n[d] = m
            image_y = numpy.zeros((len(image_array),))
            datasets = [image_array, image_y]
            save_file_name = os.path.join(self.image_path, d+".pkl") 
            f0 = open(save_file_name, 'wb')
            cPickle.dump(datasets,f0)
            f0.close()  
 
    # MERGE THE PICKLE FILE TO A FILE NAMED DATASETS.PKL
    def merge_pic(self):
        files = self.get_file_list() 
        minx = min(self.datasets_n.values())
        datasets_x = numpy.zeros((minx * len(files), self.size_x * self.size_y * self.channel))
        datasets_y = numpy.zeros((minx * len(files),),dtype='int64')
        
        for i in xrange(len(files)):
            file_name = " "
            for tem in files:
                if i == int(tem[:tem.find("-")]):
                    file_name = tem
            f = open(os.path.join(self.image_path,file_name))
            data = cPickle.load(f)
            f.close()
            index = i
            for j in xrange(len(data[0])):
                datasets_x[index,:] = data[0][j] 
                datasets_y[index]   = i
                index = index + len(files)
        return (datasets_x,datasets_y)

    def generater_datasets(self,merge_pic):
        length_sum   = merge_pic[0].shape[0]
        length_train = length_sum - int(self.valid_rate * length_sum) - \
                       int(self.test_rate * length_sum) 
        train_x = numpy.zeros((length_train,self.size_x * self.size_y * self.channel),\
                                                                       dtype='float32')
        train_y = numpy.zeros((length_train,),dtype='int64')

        length_valid = int(self.valid_rate * length_sum)
        valid_x = numpy.zeros((length_valid, self.size_x * self.size_y * self.channel),\
                                                                       dtype = 'float32')
        valid_y = numpy.zeros((length_valid,),dtype='int64')

        length_test = int(self.test_rate * length_sum)
        test_x = numpy.zeros((length_test, self.size_x * self.size_y * self.channel),\
                                                                       dtype = 'float32')
        test_y = numpy.zeros((length_test,),dtype='int64')    
 
        train_x[:][:] = merge_pic[0][:length_train][:]
        train_y[:]    = merge_pic[1][:length_train]
        valid_x[:][:] = merge_pic[0][length_train:length_train + length_valid][:]
        valid_y[:]    = merge_pic[1][length_train:length_train + length_valid]
        test_x[:][:]  = merge_pic[0][length_train + length_valid:length_train + \
                                                    length_valid + length_test][:]
        test_y[:]  = merge_pic[1][length_train + length_valid:length_train +    \
                                                 length_valid +  length_test] 
  
        datasets = ((train_x,train_y),(valid_x,valid_y),(test_x,test_y))
        #print datasets 
        
        f = open(os.path.join(self.image_path,"datasets.pkl"),"wb")
        cPickle.dump(datasets,f)
        f.close()


def DealDate_deal(image_path,size_x,size_y,channel):
    # THIS FUNCTION IS THE CONTROLLER, IT WILL INVOKE THE OTHER FUNCTION TO FINISH
    # THE TASK.
    instance = DealDate(image_path,size_x,size_y,channel)
    instance.remove_pkl_list()
    instance.convert_pic()
    tem = instance.merge_pic()
    instance.generater_datasets(tem)
        
if __name__ == '__main__':
    image_path = "/home/gd/ensemble-nn/data/image"
    size_x = 28
    size_y = 28
    channel = 3
    instance = DealDate(image_path,size_x,size_y,channel)
    instance.convert_pic()
    xx = instance.merge_pic()
    instance.generater_datasets(xx)

    
