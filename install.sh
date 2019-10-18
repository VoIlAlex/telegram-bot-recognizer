
# Download weights of NN 
cd data
wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-spp.cfg -O yolov3.cfg
wget https://pjreddie.com/media/files/yolov3-spp.weights -O yolov3.weights
cd ..


# Install python dependencies 
pip3 install -r requirements.txt