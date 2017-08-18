Adaboost-NN is a fast deep learning framework. It is developed by Gordon,a postgraduate
student.                                                                               
                                                                  
This framework is integrated by different model. Users can select the neural network algrithm(MLP, DBN, SDA, CNN) to intergrate it into a whole modle using the adaboost.m1 or adaboost.m2. All works just can be done by the config file.                                         

INSTALL:                                                                               
----python2.7                                                                          
----theano: A package of python                                                        
----GPU   : It is not necessary.But it will speed up your project if you have one.                                                                                   
note: The environment install and the tutorial of theano, you can access the webset: http://deeplearning.net/software/theano/                                                     


RUN:                                                                                   
After you download the code form github, you can see a file named ensemb.py.
It likes the main function in C++ or other program language. You should run
the command according follow format.                                                    
            

                    
command: python ensemb.py --help                                                        
you can get the result:                                                                 

----------------------------------------------------------------------------------------------------------------.                      

images_deal : deal the image to pickle file                            
----------    ensemb.py --images_deal                                  
----------              --source_path=/home/gd/ensemble-nn/data/image  
----------              --size_x=28                                    
----------              --size_y=28                                    
----------              --channel=3                                    
-----------------------------------------------------------------------------------------------------------------.                         

samples     : sample the samples from the datasets                     
----------    ensemb.py --samples                                      
----------              --no_replace=1                                 
----------              --replace=0                                    
----------              --source_path=./data/mnist.pkl.gz              
----------              --number=30000                                 
----------              --save_path=./data/mnist-sample-replace.pkl    
------------------------------------------------------------------------------------------------------------------- .                             

train_test  : train and test the model                                 
----------    ensemb.py --train_test                                   
----------              --dataset =/home/gd/ensemble-nn/data/mnist.pkl.gz                                               
----------              --config_path=./config-example                 
----------              --n_y =10                                      
----------              --m1  = 1                                      
---------------------------------------------------------------------------------------------------------------- . 
         

                          
command: python ensemb.py --train_test --dataset /home/gd/ensemble-nn/data/mnist.pkl.gz --config_path ./config-example --n_y 10 --m1 1    
                                                                                                    
you can get the result after seleral iterate.           
               
----------------------------------------------------------------------------------------------------------------.
                                                                                                                                        
Boost error :0.302462135563                                            
----Model 0 weight value :5.50250406656                                
----Model 1 weight value :2.31487073939                                
----Model 2 weight value :0.457783219979                               
----Model 3 weight value :0.835600700207                               
----The error rate of the testdatas:                                   
[0.9826, 0.9861, 0.9868, 0.9881]                                       
****************************************************************************************************************.

                                                                       
the detail message, you can read in other file.                        
                                                                       
IMPORTANT NOTE:                                                        
THIS FRAMEWORE IS DEVELOPED FOR LEARNING ADABOOST AND NEURAL NETWORK.  
                                                                       
Please just cite it in your publications if it helps your research: Author = {Gordon},Year = {2015}   
                                     
In addition, I believe this framework may be have many bugs because it is the 
first vection and just have a simple test. In the free time, I will make it more
and more strong. If you have any question, just sent to me. I promise you will
get my answer in 36 hours.                                             






