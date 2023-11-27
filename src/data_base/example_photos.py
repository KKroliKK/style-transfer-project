import os
import sqlite3 as sq

from create_bot import bot


def start():
    global base, cur
    base = sq.connect(os.path.join("data", "databases", "example_photos.sqlite"))
    cur = base.cursor()
    base.execute(
        "CREATE TABLE IF NOT EXISTS example(id INTEGER PRIMARY KEY AUTOINCREMENT, img TEXT)"
    )
    base.commit()


async def add(state):
    async with state.proxy() as data:
        cur.execute("INSERT INTO example (img) VALUES (?)", tuple(data.values()))
        base.commit()


async def read(message):
    for ret in cur.execute("SELECT * FROM example").fetchall()[:3]:
        await bot.send_photo(message.from_user.id, ret[1])


async def get_cnn2_styles(message):
    styles_photo = cur.execute("SELECT * FROM example").fetchall()[3]
    await bot.send_photo(message.from_user.id, styles_photo[1])


async def delete():
    cur.execute("DELETE FROM example")
    base.commit()
