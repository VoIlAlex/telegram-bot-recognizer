FROM hdgigante/python-opencv:4.7.0-ubuntu

WORKDIR /app

COPY data ./data
COPY install.sh .
RUN sh install.sh

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY run_bot.py .
COPY yolo_coco_net.py .
