import ctypes
import os


class Gate(object):
    def __init__(self):
        self.handle_type = ctypes.c_void_p
        # get the path of this python file
        current_file_path = os.path.abspath(__file__)
        # get the directory containing the script
        lib_path = os.path.dirname(current_file_path) + "/../../cmake-build-debug/libNeonPy/liblibNeonPy.so"
        # move up two folders with respec to script_dir

        try:
            self.lib =    ctypes.CDLL(lib_path)
        except Exception as e:
            print(f"Failed to load library: {lib_path}")
            raise e

