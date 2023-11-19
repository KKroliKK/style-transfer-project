from aiogram.utils import executor
from handlers import client, admin
from create_bot import dp
import data_base


async def on_startup(_):
    data_base.style_photos.start()
    data_base.example_photos.start()


executor.start_polling(dp, skip_updates=True, on_startup=on_startup)