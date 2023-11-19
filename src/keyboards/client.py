from aiogram.types import ReplyKeyboardMarkup, KeyboardButton


b1 = KeyboardButton('/process_photo')
b2 = KeyboardButton('/help')


menu = ReplyKeyboardMarkup(resize_keyboard=True).add(b1).add(b2)