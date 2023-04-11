import telebot
import os
import cv2
from yolo_coco_net import YoloCocoNet
import argparse
import random


TOKEN = os.getenv("TG_TOKEN")
bot = telebot.TeleBot(TOKEN)
net = YoloCocoNet('data')


class ReplyCompositor:
    def __init__(self):
        # Replies are splitted
        # into 10 levels of confidence:
        #
        # Level 1 - 90-100 %
        # Level 2 - 80-90 %
        # Level 3 - 70-80 %
        # ...
        #
        # ! Fill free to expand the levels
        # ! with phrases you think describe
        # ! the level the best way
        #
        # The first missed part
        # is an article (a, an)
        # The second messed part
        # is the class name of the
        # object
        self.reply_level_1 = [
            "I'm completely sure it's {} {}.",
            "It's {} {}. I'm sure.",
            "I'm 100% sure it's {} {}."
        ]
        self.reply_level_2 = [
            "Pretty sure it's {} {}.",
            "It's {} {}."
        ]
        self.reply_level_3 = [
            "I see {} {}."
        ]
        self.reply_level_4 = [
            "It's {} {}, isn't it?"
        ]
        self.reply_level_5 = [
            "It looks like {} {}."
        ]
        self.reply_level_6 = [
            "Hmmm. Is it a {} {}?"
        ]
        self.reply_level_7 = [
            "I'm unsure but it's pretty similar to {} {}."
        ]
        self.reply_level_8 = [
            "So can hardly see anything... Wait... It's {} {}... or a banana..."
        ]
        self.reply_level_9 = [
            "I could be anything. {} {}, a dinosaur, a lake. I don't know. Really."
        ]
        self.reply_level_10 = [
            "There is nothing on the image. Or at lease I can't see anything."
        ]

    def format_reply(self, class_name, confidence):
        assert 0.0 <= confidence <= 1.0
        confidence_level = (100 - confidence * 100) // 10 + 1
        appropriate_templates = getattr(
            self, 'reply_level_{}'.format(int(confidence_level)))
        template = random.choice(appropriate_templates)
        # choose a article
        article = 'an' if class_name[0] in 'aeiou' else 'a'
        return template.format(article, class_name)


reply_compositor = ReplyCompositor()


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
    """

    Arguments:
        objects {dict} -- class_name -> confidence

    Returns:
        str -- reply
    """
    most_probable_class_name = None
    higher_confidence = None

    for class_name, confidence in objects.items():

        # initial class name
        if most_probable_class_name is None:
            most_probable_class_name = class_name
            higher_confidence = confidence

        elif confidence > higher_confidence:
            most_probable_class_name = class_name
            higher_confidence = confidence

    reply = reply_compositor.format_reply(
        most_probable_class_name, higher_confidence)
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


@bot.message_handler(commands=["start"])
def reply_to_start(message: telebot.types.Message):
    bot.send_message(
        message.chat.id,
        "Hello! Welcome to Recognizer bot.\n"
        "Send me some image and I'll guess what is it."
    )


if __name__ == "__main__":
    try:
        bot.remove_webhook()
    except: ...
    bot.polling()
