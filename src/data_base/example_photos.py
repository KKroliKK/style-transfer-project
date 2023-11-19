from create_bot import bot
import sqlite3 as sq


def start():
    global base, cur
    base = sq.connect('example_photos.sqlite')
    cur = base.cursor()
    base.execute('CREATE TABLE IF NOT EXISTS example(id INTEGER PRIMARY KEY AUTOINCREMENT, img TEXT)')
    base.commit()


async def add(state):
    async with state.proxy() as data:
        cur.execute('INSERT INTO example (img) VALUES (?)', tuple(data.values()))
        base.commit()


async def read(message):
    for ret in cur.execute('SELECT * FROM example').fetchall():
        await bot.send_photo(message.from_user.id, ret[1])


async def delete():
    cur.execute('DELETE FROM example')
    base.commit()