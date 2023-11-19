from aiogram.types import KeyboardButton, ReplyKeyboardMarkup

b1 = KeyboardButton("/process_photo")
b2 = KeyboardButton("/help")


menu = ReplyKeyboardMarkup(resize_keyboard=True).add(b1).add(b2)
