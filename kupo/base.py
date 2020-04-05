import queue
import threading

from kupo.consts import STOP_FLAG


class AsyncModule(object):
    def __init__(self, name, manager=None):
        self.name = name
        self.queue = queue.Queue()
        self.active = True
        self.manager = None
        self.register_manager(manager)
        self.process = threading.Thread(target=self.run)

    def send(self, msg):
        self.queue.put(msg)

    def register_manager(self, manager):
        self.manager = manager

    def start(self):
        assert self.manager is not None
        self.active = True
        self.process.start()

    def stop(self):
        self.active = False
        self.queue.join()

    def run(self):
        while self.active:
            try:
                msg = self.queue.get()
            except queue.Empty:
                continue
            if type(msg) == str and msg == STOP_FLAG:
                self.queue.task_done()
                self.stop()
                break

            self.process_msg(msg)
            self.queue.task_done()

    def process_msg(self, msg):
        raise NotImplementedError()


class AudioModule(AsyncModule):
    def process_msg(self, msg):
        self.process_audio(msg)

    def process_audio(self, inp_buf):
        # Could do type check here
        raise NotImplementedError()


class ModuleManager(object):
    def __init__(self):
        self.queue = queue.Queue()
        self.context_modules = {}
        self.active = True
        self.process = threading.Thread(target=self.run)

    def register_module(self, module):
        assert module.name not in self.context_modules
        self.context_modules[module.name] = module
        module.register_manager(self)

    def send(self, msg):
        self.queue.put(msg)

    def start(self):
        self.active = True
        for module in self.context_modules.values():
            module.start()
        self.process.start()

    def stop(self):
        for m in self.context_modules.values():
            m.send(STOP_FLAG)
        self.active = False
        self.queue.join()
        for m in self.context_modules.values():
            m.process.join()

    def run(self):
        while self.active:
            try:
                msg = self.queue.get()
            except queue.Empty:
                continue

            if type(msg) == str and msg == STOP_FLAG:
                self.queue.task_done()
                self.stop()
                break

            context, module_msg = msg
            if context not in self.context_modules:
                raise ValueError("Invalid context module: {}".format(context))
            self.context_modules[context].queue.put(msg)
            self.queue.task_done()

