# -* - coding: UTF-8 -* -
import ConfigParser
import os
import sys
import numpy
import os.path
import cPickle 
import os.path 
import time,  datetime


#os.listdir(pfile):
#os.path.join(pfile,  file) 
#os.path.isfile(targetFile): 
#os.remove(targetFile)

def read_config_name(path):
    try:
        files = os.listdir(path)
        output = {}
        
        for tem in files:
            if tem[-4:] == ".cfg":
                 head = tem.find("_")
                 end  = tem.find(".cfg")
                 num  = tem[ : head]
                 kind = tem[head+1:end]
                 name = tem[:end]
                 output[num] = [kind, name, tem]
    except Exception as err:
        print err
    # the output is like the following:
    # {'1': ['dbn', '1_dbn', '1_dbn.cfg'], '0': ['sda', '0_sda', '0_sda.cfg'], 
    # '3': ['mlp', '3_mlp', '3_mlp.cfg'], '2': ['cnn', '2_cnn', '2_cnn.cfg']}
    return output

# Remove the pikcle file
def remove_pkl_file(path):
    try:
        tems = os.listdir(path)
        files = []
        for f in tems:
            if os.path.isfile(os.path.join(path,f)) and \
                f[-4:] == ".pkl":
                os.remove(os.path.join(path,f))
    except Exception as err:
        print err

def read_pkl_name(path):
    try:
        files = os.listdir(path)
        output = {}
        
        for tem in files:
            if tem[-4:] == ".pkl":
                 head = tem.find("_")
                 end  = tem.find(".pkl")
                 num  = tem[ : head]
                 kind = tem[head+1:end]
                 name = tem[:end]
                 output[num] = [kind, name, tem]
    except Exception as err:
        print err
    #print output
    # the output is like the following:
    # {'1': ['dbn', '1_dbn', '1_dbn.cfg'], '0': ['sda', '0_sda', '0_sda.cfg'], 
    # '3': ['mlp', '3_mlp', '3_mlp.cfg'], '2': ['cnn', '2_cnn', '2_cnn.cfg']}
    return output

def write(path):
    conf = ConfigParser.ConfigParser()
    conf.add_section('input_layer')
    conf.set('input_layer','input_num',28*28)
    
    conf.add_section('hidden_layer_0')
    conf.set('hidden_layer_0','hidden_num',100)

    conf.add_section('lr_layer')
    conf.set('lr_layer','lr_num',10)

    conf.add_section('parm')
    conf.set('parm','learning_rate',0.01)
    conf.set('parm','l1_reg',0.00)
    conf.set('parm','l2_reg',0.0001)
    conf.set('parm','n_epochs',10)
    conf.set('parm','batch_size',100)

    conf.write(open('./config-example/model_mlp_0.cfg','w'))

def read_parser(path):
    conf = ConfigParser.ConfigParser()

    # read the config object using the path.
    conf.read(path)

    # get the section list.
    sections = conf.sections()

    # get the parm dictionary from the path file.
    con = {}
    for section in sections:
        tem_sec = {}
        tem_key = {}
        for key in conf.options(section):
            tem_key[key] = conf.get(section,key)
        con[section] = tem_key
    return con
    """   
    # The follow is the interface of pasering the file. 
    #     得到指定section的所有option
    #     options = conf.options("sec_a")
    #     print 'options:', options       
    #     得到指定section的所有键值对
    #     kvs = conf.items("sec_a")
    #     print 'sec_a:', kvs                
    #     指定section，option读取值
    #     str_val = conf.get("sec_a", "a_key1")
    #     int_val = conf.getint("sec_a", "a_key2")
    """

#read_config_name("../config-example")
