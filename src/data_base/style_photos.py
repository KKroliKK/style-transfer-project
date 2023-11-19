from create_bot import bot
import sqlite3 as sq


def start():
    global base, cur
    base = sq.connect('style_photos.sqlite')
    cur = base.cursor()
    base.execute('CREATE TABLE IF NOT EXISTS style(id INTEGER PRIMARY KEY AUTOINCREMENT, img TEXT)')
    base.commit()


async def add(state):
    async with state.proxy() as data:
        cur.execute('INSERT INTO style (img) VALUES (?)', tuple(data.values()))
        base.commit()


async def read(message):
    for row in cur.execute('SELECT * FROM style').fetchall():
        await bot.send_photo(message.from_user.id, row[1])


async def read2():
    return cur.execute('SELECT * FROM style').fetchall()


async def get(data):
    return cur.execute('SELECT img FROM style WHERE id == ?', data).fetchone()


async def delete(data):
    cur.execute('DELETE FROM style WHERE id == ?', (data,))
    base.commit()