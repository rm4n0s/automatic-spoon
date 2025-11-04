import multiprocessing
import asyncio
from src import models, config


class Manager:
    def __init__(self, conf: config.Config, view_queue: multiprocessing.Queue[str]):
        self.queue = view_queue
        self.config = config


def start_manager(conf: config.Config, view_queue: multiprocessing.Queue[str]):
    manager = Manager(conf, view_queue)
