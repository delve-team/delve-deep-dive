import telegram
from telegram.ext import Updater
from telegram.ext import CommandHandler
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple Bot to reply to Telegram messages.

This is built on the API wrapper, see echobot2.py to see the same example built
on the telegram.ext bot framework.
This program is dedicated to the public domain under the CC0 license.
"""
import logging
import telegram
from telegram.error import NetworkError, Unauthorized
from time import sleep
import argparse
import os
from telegram_runner import telegram_worker

parser = argparse.ArgumentParser(description='Train a network on a dataset')
parser.add_argument('-n', '--network', dest='model_name', action='store', default='vgg11')
parser.add_argument('-d', '--dataset', dest='dataset_name', action='store', default='Cifar10')
parser.add_argument('-b', '--batch-size', dest='batch_size', action='store', default=32)
parser.add_argument('-o', '--output', dest='output', action='store', default='logs')
parser.add_argument('-c', '--compute-device', dest='device', action='store', default='cpu')
parser.add_argument('-r', '--run_id', dest='run_id', action='store', default=0)
parser.add_argument('-cf', '--config', dest='json_file', action='store', default=None)

update_id = None


def main():
    """Run the bot."""

    credentials = ''
    with open('credentials', 'r') as fp:
        credentials = fp.read()
    global update_id
    # Telegram Bot Authorization Token
    bot = telegram.Bot(credentials)

    # get the first pending update_id, this is so we can skip over it in case
    # we get an "Unauthorized" exception.
    try:
        update_id = bot.get_updates()[0].update_id
    except IndexError:
        update_id = None

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    echo(bot, flush=True)
    while True:
        try:
            echo(bot)
        except NetworkError:
            sleep(1)
        except Unauthorized:
            # The user has removed or blocked the bot.
            update_id += 1


def exec_msg(message):
    raw_text = message.text
    splitted = raw_text.split(' ')
    print(splitted)
    #try:
    if True:
        if splitted[0] == 'run':
            message.reply_text('running command received, executing experiments')
            print('Executing experiments')
            telegram_worker(message.reply_text, raw_text[4:])

        elif 'execute experiment' in raw_text:
            if 'cd_batch' in raw_text:
                ds = 'catdog'
            elif 'cf10_batch' in raw_text:
                ds = 'cifar10'
            elif 'cf100_batch' in raw_text:
                ds = 'cifar100'
            command = '--config ./configs/block_a_{}.json --run_id 0 -c cuda:0'.format(ds)
            telegram_worker(message.reply_text, command)


        else:
            message.reply_text('unparseable command')
    #except:
    #        message.reply_text('Execution has failed')
    print('Execution has finished')
    return





def echo(bot, flush=False):
    """Echo the message the user sent."""
    global update_id
    # Request updates after the last update_id
    for update in bot.get_updates(offset=update_id, timeout=10):
        update_id = update.update_id + 1

        if update.message:  # your bot can receive updates without messages
            # Reply to the message
            print(update.message.text)
            if not flush:
                exec_msg(update.message)


if __name__ == '__main__':
    main()