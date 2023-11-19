from aiogram.types import ReplyKeyboardMarkup, KeyboardButton


menu = ReplyKeyboardMarkup(resize_keyboard=True)\
            .row(KeyboardButton('/view_resolution'), KeyboardButton('/change_resolution'))\
            .row(KeyboardButton('/view_styles'), KeyboardButton('/add_style'), KeyboardButton('/del_style'))\
            .row(KeyboardButton('/view_exampl'), KeyboardButton('/add_example'), KeyboardButton('/del_example'))\
            .row(KeyboardButton('/cancel'), KeyboardButton('/client'))