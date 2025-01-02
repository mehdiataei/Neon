import ctypes
from enum import Enum
from typing import List
import warp as wp

import numpy as np

# from neon import neon


class Report(object):
    def __init__(self, report_name: str = 'report'):

        self.handle: ctypes.c_void_p = ctypes.c_void_p(0)
        self.report_name = c_string = report_name.encode('utf-8')

        try:
            self.neon: neon = neon()
        except Exception as e:
            self.handle = ctypes.c_void_p(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self.help_load_api()
        self.help_report_new()

    def __del__(self):
        if self.backend_handle == 0:
            return
        self.help_backend_delete()
        pass

    def help_load_api(self):
        self.api = {}

        def register_foo(name, argtypes, restype):
            foo = getattr(self.neon.lib, name, None)
            # add a class variable with the name stored in the name variable
            foo.argtypes = argtypes
            foo.restype = restype
            self.api[name] = foo
            pass

        # ------------------------------------------------------------------
        register_foo('report_new',
                     [ctypes.POINTER(self.neon.handle_type), ctypes.c_char_p],
                     ctypes.c_int)
        # ------------------------------------------------------------------
        register_foo('report_delete',
                     [ctypes.POINTER(self.neon.handle_type)],
                     ctypes.c_int)
        # ------------------------------------------------------------------
        for type in [ctypes.c_int32,
                     ctypes.c_int64,
                     ctypes.c_uint32,
                     ctypes.c_float32,
                     ctypes.c_float64]:
            register_foo('report_add_member',
                         [
                             ctypes.POINTER(self.neon.handle_type),
                             ctypes.c_char_p,
                             type],
                         ctypes.c_int)
        # ------------------------------------------------------------------
        for type in [ctypes.c_int32,
                     ctypes.c_int64,
                     ctypes.c_uint32,
                     ctypes.c_float32,
                     ctypes.c_float64]:
            register_foo('report_add_member_vector',
                         [
                             ctypes.POINTER(self.neon.handle_type),
                             ctypes.c_char_p,
                             ctypes.POINTER(type)],
                         ctypes.c_int)
            # ------------------------------------------------------------------
            register_foo('report_write',
                         [
                             ctypes.POINTER(self.neon.handle_type),
                             ctypes.c_char_p,
                             ctypes.c_bool],
                         ctypes.c_int)
        # ------------------------------------------------------------------

    def help_new(self):
        ret = self.neon.lib.report_new(self.handle, self.report_name)
        if ret != 0:
            # raise excetion

            pass

    def help_backend_delete(self):
        if self.backend_handle == 0:
            return
        # print(f'PYTHON cuda_driver_handle {hex(self.cuda_driver_handle.value)}')
        self.neon.lib.cuda_driver_delete(ctypes.pointer(self.cuda_driver_handle))
        # print(f'PYTHON backend_handle {hex(self.backend_handle.value)}')
        res = self.neon.lib.dBackend_delete(ctypes.pointer(self.backend_handle))
        if res != 0:
            raise Exception('Failed to delete backend')

    def get_num_devices(self):
        return self.n_dev

    def get_warp_device_name(self):
        if self.runtime == Backend.Runtime.stream:
            return 'cuda'
        else:
            return 'cpu'

    def __str__(self):
        return ctypes.cast(self.neon.lib.get_string(self.backend_handle), ctypes.c_char_p).value.decode('utf-8')

    def sync(self):
        return self.neon.lib.dBackend_sync(self.backend_handle)

    def get_device_name(self, dev_idx: int):
        if self.runtime == Backend.Runtime.stream:
            dev_id = self.dev_idx_list[dev_idx]
            return f"cuda:{dev_id}"
        else:
            dev_id = self.dev_idx_list[dev_idx]
            return f"cpu:{dev_id}"
