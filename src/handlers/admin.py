from create_bot import dp, bot
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher.filters import Text
from aiogram import types
from data_base import example_photos, style_photos
from keyboards.admin import menu
import keyboards
import json
from data_base import style_photos
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from create_bot import ID_ADMIN


@dp.message_handler(commands='admin')
async def admin_mod(message: types.Message):
    if message.from_user.id == ID_ADMIN:
        await bot.send_message(message.from_user.id, 'Admin mode is enabled!', reply_markup=menu)


@dp.message_handler(commands='client')
async def admin_mod(message: types.Message):
    if message.from_user.id == ID_ADMIN:
        await bot.send_message(message.from_user.id, 'Client mode is enabled!', reply_markup=keyboards.client.menu)


@dp.message_handler(state='*', commands='cancel')
@dp.message_handler(Text(equals='cancel', ignore_case=True), state='*')
async def cancel_handler(message: types.Message, state: FSMContext):
    if message.from_user.id == ID_ADMIN:
        current_state = await state.get_state()
        if current_state is None:
            return
        await state.finish()
        await message.reply('ok')
    



# Add photos with styles to Bot's data base ---------------------------------------
class FSMStyle(StatesGroup):
    photo = State()


@dp.message_handler(commands='add_style', state=None)
async def add_style(message: types.Message):
    if message.from_user.id == ID_ADMIN:
        await FSMStyle.photo.set()
        await message.reply('Download photo')


@dp.message_handler(content_types=['photo'], state=FSMStyle.photo)
async def load_photo(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['img'] = message.photo[-1].file_id
    await style_photos.add(state)
    await state.finish()



@dp.message_handler(commands='delete_style')
async def delete_item(message: types.Message):
    if message.from_user.id == ID_ADMIN:
        read = await style_photos.read2()
        for record in read:
            await bot.send_photo(message.from_user.id, record[1], 
                                    reply_markup=InlineKeyboardMarkup().\
                                       add(InlineKeyboardButton(f'Delete', callback_data=f'del {record[0]}')))


@dp.callback_query_handler(lambda x: x.data and x.data.startswith('del '))
async def del_callback_run(callback_query: types.CallbackQuery):
    await style_photos.delete(callback_query.data.replace('del ', ''))
    await callback_query.answer(text=f'{callback_query.data.replace("del ", "")} deleted.', show_alert=False)
    await bot.answer_callback_query(callback_query.id, text=f'{callback_query.data.replace("del ", "")} deleted.')


@dp.message_handler(commands=['view_styles'])
async def view_styles(message : types.Message):
    if message.from_user.id == ID_ADMIN:
	    await style_photos.read(message)




# Add example photos to Bot's data base -----------------------------
class FSMExample(StatesGroup):
    photo = State()


@dp.message_handler(commands='add_example', state=None)
async def add_style(message: types.Message):
    if message.from_user.id == ID_ADMIN:
        await FSMExample.photo.set()
        await message.reply('Download photo')


@dp.message_handler(content_types=['photo'], state=FSMExample.photo)
async def load_photo(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['img'] = message.photo[-1].file_id
    await example_photos.add(state)
    await state.finish()


@dp.message_handler(commands=['view_exampl'])
async def view_example(message : types.Message):
    if message.from_user.id == ID_ADMIN:
	    await example_photos.read(message)


@dp.message_handler(commands=['delete_example'])
async def delete_example(message : types.Message):
    if message.from_user.id == ID_ADMIN:
        await example_photos.delete()
        await message.reply('Deleted!')



# Change resolution of processed photos -----------------------------

@dp.message_handler(commands='view_resolution')
async def add_style(message: types.Message):
    if message.from_user.id == ID_ADMIN:
        with open('style_transfer/resolution.json') as json_file:
            data = json.load(json_file)
            resolution = data['resolution']
        await message.reply(str(resolution))


class FSMResolution(StatesGroup):
    resolution = State()


@dp.message_handler(commands='change_resolution', state=None)
async def add_style(message: types.Message):
    if message.from_user.id == ID_ADMIN:
        await FSMResolution.resolution.set()
        await bot.send_message(message.from_user.id, 'Enter new resolution:')


@dp.message_handler(content_types=['text'], state=FSMResolution.resolution)
async def load_photo(message: types.Message, state: FSMContext):
    if message.from_user.id == ID_ADMIN:
        with open('style_transfer/resolution.json', 'w') as json_file:
                json.dump({"resolution": int(message.text)}, json_file)
        await message.reply('Done!')
        await state.finish()
