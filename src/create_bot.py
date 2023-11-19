import os
import logging
import json
from aiogram import Bot
from aiogram.dispatcher import Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from dotenv import load_dotenv


storage = MemoryStorage()

with open("key_data.json") as json_file:
    data = json.load(json_file)
    API_TOKEN = data['API_TOKEN']
    ID_ADMIN = data['ID_ADMIN']

DOWNLOAD_URL = 'https://api.telegram.org/file/bot' + API_TOKEN + '/'

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot, storage=storage)
