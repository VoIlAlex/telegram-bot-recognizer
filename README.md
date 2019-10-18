# telegram-bot-recognizer

## Installation

First of all install python 3.x with pip.

#### Auto-installation

```
sh install.sh
```

#### Manual installation

1. Install python dependencies:

```
pip3 install -r requirements.txt
```

2. To run this bot you should download weights and configuration file at [darknet](https://pjreddie.com/darknet/yolo/).

Place this file as `data/yolov3.cfg` - [configuration file](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-spp.cfg)
Place this file as `data/yolov3.weights` - [weights file](https://pjreddie.com/media/files/yolov3-spp.weights)

## Running

To start the bot run the following command.

```
python3 run_bot.py
```

If you don't want to save images sent to server:

```
python3 run_bot.py --delete
```
