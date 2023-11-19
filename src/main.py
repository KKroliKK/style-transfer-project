from aiogram.utils import executor

from create_bot import dp
from data_base import example_photos, style_photos
from handlers import admin, client


async def on_startup(_):
    style_photos.start()
    example_photos.start()


executor.start_polling(dp, skip_updates=True, on_startup=on_startup)
