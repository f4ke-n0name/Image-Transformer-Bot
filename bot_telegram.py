import logging
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import InlineKeyboardMarkup, ReplyKeyboardMarkup
from aiogram.types import InlineKeyboardButton, KeyboardButton

import asyncio

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
from copy import deepcopy
from urllib.parse import urljoin

from style_transfer import *
from GAN import transfer

logging.basicConfig(level=logging.INFO)
bot = Bot(token="6313560891:AAEuW3-cCwTomXZeFtbB8UkOleMIovaInWc")
dp = Dispatcher(bot)

image_buffer = {}


class InfoAboutUser:
    def __init__(self):
        self.settings = {'num_epochs': 200,
                         'imgsize': 256}
        self.images = []


start_bot_kb = ReplyKeyboardMarkup()
start_bot_kb.add(KeyboardButton('/start'))
start_kb = InlineKeyboardMarkup(resize_keyboard=True)
start_kb.add(InlineKeyboardButton('Перенос стиля', callback_data='nst'))
start_kb.add(InlineKeyboardButton('Изображение в стиле Ван Гога', callback_data='van_gogh'))
start_kb.add(InlineKeyboardButton('Изображение стиле Моне', callback_data='monet'))
back_kb = InlineKeyboardMarkup()
back_kb.add(InlineKeyboardButton("Далее", callback_data='next'))
back_kb.add(InlineKeyboardButton('Назад', callback_data='menu'))


@dp.message_handler(commands=['start'])
@dp.async_task
async def cmd_start(message: types.Message):
    await message.answer("Привет, я бот для изменения стиля изображений!\n" + "Пожалуйста, выберите одну из функций:"
                                                                              "\n\t -- Перенести стиль с одного изображения на другое изображение."
                                                                              "\n\t -- Перенести один из предложенных стилей на изображение.",
                         reply_markup=start_kb)
    image_buffer[message.chat.id] = InfoAboutUser()


@dp.message_handler(commands=['help'])
async def cmd_help(message: types.Message):
    await message.answer("Привет, я Neural Style Transfer бот (NST бот).\n" +
                         "Я умею переносить стили из одного изображения на другое.\n"
                         "Вот мои возможности:", reply_markup=start_kb)


@dp.callback_query_handler(lambda x: x.data == 'menu')
async def menu(callback_query: types.callback_query):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("Выберете функцию:", reply_markup=start_kb)


@dp.callback_query_handler(lambda x: x.data == 'nst')
async def nst_fun(callback_query):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("Выбрана функция : перенос стиля")
    if callback_query.from_user.id not in image_buffer:
        image_buffer[callback_query.from_user.id] = InfoAboutUser()
    image_buffer[callback_query.from_user.id].st_type = 'nst'
    await callback_query.message.edit_reply_markup(reply_markup=back_kb)


@dp.callback_query_handler(lambda x: x.data == 'van_gogh')
async def van_gogh(callback_query):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("Выбрана функция : стилизация под Ван Гога.")
    if callback_query.from_user.id not in image_buffer:
        image_buffer[callback_query.from_user.id] = InfoAboutUser()
    image_buffer[callback_query.from_user.id].st_type = 'van_gogh'
    await callback_query.message.edit_reply_markup(reply_markup=back_kb)


@dp.callback_query_handler(lambda x: x.data == 'monet')
async def monet(callback_query):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("Выбрана функция : стилизация под Моне.")
    if callback_query.from_user.id not in image_buffer:
        image_buffer[callback_query.from_user.id] = InfoAboutUser()
    image_buffer[callback_query.from_user.id].st_type = 'monet'
    await callback_query.message.edit_reply_markup(reply_markup=back_kb)


@dp.callback_query_handler(lambda x: x.data == 'next')
async def load_images(callback_query):
    if image_buffer[callback_query.from_user.id].st_type == 'nst':
        await callback_query.message.edit_text("Пришли мне изображение, стиль с которого нужно перенести.\n"
                                               "P.S. Для лучшего качества рекомендуется отправлять файлы как .png  или .jpg .\n"
                                               "P.P.S Другие форматы не обработаются, если Вы это сделали, то напишите /start")
        image_buffer[callback_query.from_user.id].need_images = 2

    elif image_buffer[callback_query.from_user.id].st_type == 'van_gogh':
        await callback_query.message.edit_text("Пришли мне изображение, и я его обработую в стиле Ван Гога.\n"
                                               "P.S. Для лучшего качества рекомендуется отправлять файлы как .png  или .jpg .\n"
                                               "P.P.S Другие форматы не обработаются, если Вы это сделали, то напишите /start")
        image_buffer[callback_query.from_user.id].need_images = 1

    elif image_buffer[callback_query.from_user.id].st_type == 'monet':
        await callback_query.message.edit_text("Пришли мне изображение, и я его обработую в стиле Моне.\n"
                                               "P.S. Для лучшего качества рекомендуется отправлять файлы как .png  или .jpg .\n"
                                               "P.P.S Другие форматы не обработаются, если Вы это сделали, то напишите /start")
        image_buffer[callback_query.from_user.id].need_images = 1


@dp.message_handler(content_types=['photo', 'document'])
async def get_image(message):
    if message.content_type == 'photo':
        img = message.photo[-1]
    else:
        img = message.document
        if img.mime_type[:5] != 'image':
            await message.answer("Загрузите файл как изображение, пожалуйста.", reply_markup=start_kb)

    file_info = await bot.get_file(img.file_id)
    image = await bot.download_file(file_info.file_path)

    if message.chat.id not in image_buffer:
        await message.answer("Сначала выберете функцию.", reply_markup=start_kb)
        return
    image_buffer[message.chat.id].images.append(image)

    if image_buffer[message.chat.id].st_type == 'nst':
        if image_buffer[message.chat.id].need_images == 2:
            await message.answer("Клаас! А теперь пришли изображение, на которое нужно перенести этот стиль.\n"
                                  "P.S. Для лучшего качества рекомендуется отправлять файлы как .png  или .jpg .\n"
                                               "P.P.S Другие форматы не обработаются, если Вы это сделали, то напишите /start")
            image_buffer[message.chat.id].need_images = 1
        elif image_buffer[message.chat.id].need_images == 1:
            await message.answer("Отлично! Начинаю обрабатывать!\n"
                                 "Это займет около 1-2 минуты.")
            log(image_buffer[message.chat.id])
            try:
                output = await style_transfer(Simple_style_transfer, image_buffer[message.chat.id],
                                              *image_buffer[message.chat.id].images)
                await bot.send_document(message.chat.id, deepcopy(output))
                await bot.send_photo(message.chat.id, output)
            except RuntimeError as err:
                if str(err)[:19] == 'CUDA out of memory.':
                    await message.answer("Не хватает мощностей! Попробуйте позже.")
                else:
                    await message.answer("Призошла неизвестная ошибка.")
            except Exception:
                await message.answer(
                    "Произошла неизвестная ошибка. Пожалуйста, делайте так, как задумано разработчиком.")
            await message.answer("Каковы дальнейшие действия?", reply_markup=start_kb)
            del image_buffer[message.chat.id]
    elif image_buffer[message.chat.id].st_type in ['van_gogh', 'monet'] and image_buffer[
        message.chat.id].need_images == 1:
        await message.answer('Отлично! Начинаю обрабатывать!\n'
                             'Это займет около 1-2 минуты.')
        log(image_buffer[message.chat.id])
        try:
            output = gan_transfer(image_buffer[message.chat.id], image_buffer[message.chat.id].images[0])
            await bot.send_document(message.chat.id, deepcopy(output))
            await bot.send_photo(message.chat.id, output)
        except RuntimeError as err:
            if str(err)[:19] == 'CUDA out of memory.':
                await message.answer("Не хватает мощностей! Попробуйте позже.")
            else:
                await message.answer("Призошла неизвестная ошибка.")
        except Exception:
            await message.answer(
                "Произошла неизвестная ошибка. Пожалуйста, делайте так, как задумано разработчиком.")
        await message.answer("Каковы дальнейшие действия?", reply_markup=start_kb)
        del image_buffer[message.chat.id]


@dp.message_handler()
async def text(message):
    await message.answer("Пожалуйста, пользуйтесь кнопками. Для предотвращения ошибок выберете функцию.",
                         reply_markup=start_kb)


async def style_transfer(style_class, user, *images):
    style = style_class(*images, imgsize=user.settings['imgsize'],
                        num_steps=user.settings['num_epochs'],
                        style_weight=10000, content_weight=1)
    output = await style.transfer()
    return tensor_to_image(output)


def gan_transfer(user, img):
    output = transfer(img, style=user.st_type)
    return tensor_to_image(output.add(1).div(2))


def tensor_to_image(tensor):
    output = np.rollaxis(tensor.cpu().detach().numpy()[0], 0, 3)
    output = Image.fromarray(np.uint8(output * 255))

    bio = BytesIO()
    bio.name = 'result.jpg'
    output.save(bio, 'JPEG')
    bio.seek(0)

    return bio


def draw_img(img):
    plt.imshow(np.rollaxis(img.cpu().detach()[0].numpy(), 0, 3))
    plt.show()


def draw_photo(*images):
    for image in images:
        img = np.array(Image.open(image))
        plt.imshow(img)
        plt.show()


def log(user):
    print()
    print('type:', user.st_type)
    if user.st_type == 1 or user.st_type == 2:
        print('settings:', user.settings)
        print('Epochs:')
    else:
        print('settings: imgsize:', user.settings['imgsize'])


executor.start_polling(dp, skip_updates=True)
#если запускать на сервере

#async def on_startup(dispatcher):
#    logging.warning('Запуск')
#
#   await bot.set_webhook(webhook_url)
#
#
#async def on_shutdown(dispatcher):
#    logging.warning('Выключение')

#webhook_path = f'/webhook/6313560891:AAEuW3-cCwTomXZeFtbB8UkOleMIovaInWc'
#webhook_url = urljoin(<webhook_host> , webhook_path) , webhook_host вставляется
#webapp_host = '0.0.0.0'
#webapp_port = int(os.environ.get('PORT' , webapp_host)
#executor.start_webhook(dp , webhook_path , on_startup, on_shotdown , skip_updates = True, host = webapp_host , port = webapp_port)
#
#
