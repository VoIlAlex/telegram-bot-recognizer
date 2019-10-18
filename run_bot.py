import telebot
import os
import cv2
import numpy as np
from yolo_coco_net import YoloCocoNet
import argparse


TOKEN = '687390205:AAGzjP1hTwpVYhSdBnmHSE1-Hy7UeCKHG78'
bot = telebot.TeleBot(TOKEN)
net = YoloCocoNet('data')


def objects_on_image(image):
    """

    Arguments:
        image -- image for detection

    Returns:
        dict -- object->confidence
    """
    predictions = net.detect(image, 0, 0.2)
    objects = {}  # class_name -> confidence

    # Here I will omit
    # bounding boxes
    for pred in predictions:
        class_name = net.labels[pred.class_id]
        if class_name in objects:
            objects[class_name] = max(objects[class_name], pred.confidence)
        else:
            objects[class_name] = pred.confidence

    return objects


def format_reply(objects):
    # TODO: here frontend goes. Update it

    reply = '\n'.join(str(k) + ' -> ' + str(v) for k, v in objects.items())
    if reply == '':
        reply = 'There was no objects detected'
    return reply


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--delete',
        action='store_true',
        help='delete photos after reply'
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()


@bot.message_handler(content_types=['photo'])
def reply_to_image(message: telebot.types.Message):
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    user_id = message.from_user.id

    # Directory with images
    # from the user.
    dir_name = 'images/{}'.format(user_id)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    file_path = os.path.join(dir_name, file_id + '.jpg')

    # Save the received image
    with open(file_path, 'wb') as image_file:
        image_file.write(downloaded_file)

    objects = objects_on_image(cv2.imread(file_path))
    reply = format_reply(objects)
    bot.reply_to(message, reply)

    # Delete files if necessary
    try:
        if args.delete:
            os.remove(file_path)
    except:
        pass


if __name__ == "__main__":
    bot.polling()
