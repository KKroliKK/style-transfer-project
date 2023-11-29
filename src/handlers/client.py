from io import BytesIO

import requests
import torch.optim as optim
import numpy as np
from aiogram import types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import (InlineKeyboardButton, InlineKeyboardMarkup,
                           ReplyKeyboardRemove)

from create_bot import DOWNLOAD_URL, ID_ADMIN, bot, dp
from data_base import example_photos, style_photos
from keyboards.client import menu
from messages import mes
from model_cnn.utils import get_transformed_photo

start_inline = InlineKeyboardMarkup(row_width=1).add(
    InlineKeyboardButton(text=mes.see_example, callback_data="example")
)


@dp.message_handler(commands=["start"])
async def command_start(message: types.Message):
    await bot.send_message(message.from_user.id, mes.hello, reply_markup=start_inline)
    await message.delete()


@dp.callback_query_handler(Text(equals="example"))
async def cancel_callback(callback: types.CallbackQuery):
    await bot.send_message(callback.from_user.id, mes.result_example, reply_markup=menu)
    await example_photos.read(callback)


@dp.message_handler(commands=["help"])
async def command_help(message: types.Message):
    await bot.send_message(message.from_user.id, mes.help, reply_markup=menu)


class FSMProcess(StatesGroup):
    choose_model = State()
    choose_style = State()
    choose_styles_cnn2 = State()
    download_style = State()
    assign_style = State()
    load_content = State()
    load_content_cnn2 = State()


model_inline = InlineKeyboardMarkup(row_width=1).add(
    InlineKeyboardButton(text="Model 1", callback_data="model_1"),
    InlineKeyboardButton(
        text="Model 2", callback_data="model_2"
    ),
    InlineKeyboardButton(text="cancel", callback_data="cancel"),
)


@dp.message_handler(commands="process_photo", state=None)
async def choose_style(message: types.Message):
    await FSMProcess.choose_model.set()
    await bot.send_message(
        message.from_user.id, mes.model_1, reply_markup=ReplyKeyboardRemove()
    )
    await bot.send_message(message.from_user.id, mes.model_2, reply_markup=model_inline)


style_inline = InlineKeyboardMarkup(row_width=1).add(
    InlineKeyboardButton(text="Upload your own style photo", callback_data="download"),
    InlineKeyboardButton(
        text="Choose a ready-made style photo", callback_data="choose"
    ),
    InlineKeyboardButton(text="cancel", callback_data="cancel"),
)


@dp.callback_query_handler(Text(equals="model_1"), state=FSMProcess.choose_model)
async def choose_style(callback: types.CallbackQuery, state: FSMContext):
    await callback.answer("1-st model is chosen")
    await FSMProcess.choose_style.set()
    await bot.send_message(
        callback.from_user.id, mes.style_1, reply_markup=ReplyKeyboardRemove()
    )
    await bot.send_message(callback.from_user.id, mes.style_2, reply_markup=style_inline)


@dp.callback_query_handler(Text(equals="cancel"), state="*")  # FSMProcess.choose_style)
async def cancel_callback(callback: types.CallbackQuery, state: FSMContext):
    await state.finish()
    await bot.send_message(callback.from_user.id, mes.cancel, reply_markup=menu)
    await callback.answer(text="Canceled")


cancel_inline = InlineKeyboardMarkup(row_width=1).add(
    InlineKeyboardButton(text="cancel", callback_data="cancel")
)


@dp.callback_query_handler(Text(equals="download"), state=FSMProcess.choose_style)
async def offer_send_style(callback: types.CallbackQuery):
    await FSMProcess.download_style.set()
    await bot.send_message(
        callback.from_user.id, mes.download_style, reply_markup=cancel_inline
    )
    await callback.answer(text="Waiting for your style photo")


@dp.message_handler(content_types=["photo"], state=FSMProcess.download_style)
async def dowload_style_photo(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data["style"] = message.photo[-1].file_id
    await FSMProcess.load_content.set()
    await bot.send_message(
        message.from_user.id, mes.download_content, reply_markup=cancel_inline
    )


@dp.callback_query_handler(Text(equals="choose"), state=FSMProcess.choose_style)
async def offer_choose_style(callback: types.CallbackQuery):
    await FSMProcess.assign_style.set()
    await callback.answer(text="Waiting for you to shoose a style")
    for photo in await style_photos.read2():
        await bot.send_photo(
            callback.from_user.id,
            photo[1],
            reply_markup=InlineKeyboardMarkup().add(
                InlineKeyboardButton("Choose", callback_data=f"index: {photo[0]}")
            ),
        )
    await bot.send_message(
        callback.from_user.id, mes.choose_style, reply_markup=cancel_inline
    )


@dp.callback_query_handler(Text(startswith="index: "), state=FSMProcess.assign_style)
async def save_choosed_style(callback: types.CallbackQuery, state: FSMContext):
    await FSMProcess.load_content.set()
    await callback.answer(text="The choise is accepted")
    async with state.proxy() as data:
        style = await style_photos.get(callback.data.replace("index: ", ""))
        data["style"] = style[0]
    await bot.send_message(
        callback.from_user.id, mes.download_content, reply_markup=cancel_inline
    )


@dp.message_handler(content_types=["photo"], state=FSMProcess.load_content)
async def dowload_content_photo(message: types.Message, state: FSMContext):
    await bot.send_message(
        message.from_user.id, mes.wait_result, reply_markup=ReplyKeyboardRemove()
    )

    style_id = None
    async with state.proxy() as data:
        style_id = data["style"]

    style = await bot.get_file(style_id)
    style = BytesIO(requests.get(DOWNLOAD_URL + style.file_path, stream=True).content)

    content = await bot.get_file(message.photo[-1].file_id)
    content = BytesIO(
        requests.get(DOWNLOAD_URL + content.file_path, stream=True).content
    )

    await bot.send_message(message.from_user.id, mes.result, reply_markup=menu)

    result1 = await get_transformed_photo(style, content, optimizer=optim.LBFGS)
    result_mes = await bot.send_photo(message.from_user.id, result1)
    # await bot.send_photo(ID_ADMIN, style_id)
    # await bot.send_photo(ID_ADMIN, message.photo[-1].file_id)
    if message.from_user.id != ID_ADMIN:
        await bot.send_photo(ID_ADMIN, result_mes.photo[-1].file_id)

    result2 = await get_transformed_photo(
        style, content, content_weight=30, optimizer=optim.Adagrad
    )
    result_mes = await bot.send_photo(message.from_user.id, result2)

    await state.finish()


# model 2

@dp.callback_query_handler(Text(equals="model_2"), state=FSMProcess.choose_model)
async def choose_style(callback: types.CallbackQuery, state: FSMContext):
    await callback.answer("2-nd model is chosen")
    await FSMProcess.choose_styles_cnn2.set()
    await example_photos.get_cnn2_styles(callback)
    await bot.send_message(callback.from_user.id, mes.style_proportions, reply_markup=cancel_inline)


@dp.message_handler(state=FSMProcess.choose_styles_cnn2)
async def choose_style(message: types.Message, state: FSMContext):
    try:
        array = np.array([int(x) for x in message.text.split()])
        assert ((array <= 7) & (array >= 0)).sum() == len(array)
        ratio = 1 / len(array)
        styles_dict = {index: ratio for index in array}
        async with state.proxy() as data:
            data["styles_dict"] = styles_dict
    except:
        await bot.send_message(message.from_user.id, "Please try again", reply_markup=cancel_inline)
        return
    
    await bot.send_message(message.from_user.id, mes.content_photo_cnn2, reply_markup=cancel_inline)
    await FSMProcess.load_content_cnn2.set()


from model_cnn_2.inference import call_cnn2

@dp.message_handler(content_types=["photo"], state=FSMProcess.load_content_cnn2)
async def upload_content_photo(message: types.Message, state: FSMContext):
    await bot.send_message(
        message.from_user.id, mes.wait_result_2, reply_markup=ReplyKeyboardRemove()
    )

    async with state.proxy() as data:
        styles_dict = data["styles_dict"]

    content = await bot.get_file(message.photo[-1].file_id)
    content = BytesIO(
        requests.get(DOWNLOAD_URL + content.file_path, stream=True).content
    )

    result = await call_cnn2(content=content, style_index=styles_dict)
    result_mes = await bot.send_photo(message.from_user.id, result, reply_markup=menu)

    if message.from_user.id != ID_ADMIN:
        await bot.send_photo(ID_ADMIN, result_mes.photo[-1].file_id)

    await state.finish()