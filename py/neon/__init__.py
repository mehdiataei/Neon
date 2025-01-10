import copy
import ctypes
from enum import Enum
import os
import warp as wp

# from .py_ne import neon
from .gate import Gate
from .dataview import DataView
from .execution import Execution
from .index_3d import Index_3d
from .ngh_idx import Ngh_idx

from .tool.__init__ import *
from .dense.__init__ import *
from .loader import Loader
from .container import Container
from .timer import Timer
from .skeleton import Skeleton

#from .block.__init__ import *
#from .multires.__init__ import *



#
#
# class PyNeon(object):
#     def __init__(self):
#         self.handle_type = ctypes.POINTER(ctypes.c_uint64)
#         self.lib = ctypes.CDLL(
#             '/home/max/repos/neon/warp/neon_warp_testing/neon_py_bindings/cmake-build-debug/libNeonPy/liblibNeonPy.so')
#         # # grid_new
#         # self.lib.grid_new.argtypes = [self.handle_type]
#         # # self.lib.grid_new.re = [ctypes.c_int]
#         # # grid_delete
#         # self.lib.grid_delete.argtypes = [self.handle_type]
#         # # self.lib.grid_delete.restype = [ctypes.c_int]
#         # # new_field
#         # self.lib.field_new.argtypes = [self.handle_type, self.handle_type]
#         # # delete_field
#         # self.lib.field_delete.argtypes = [self.handle_type]
#
#     def field_new(self, handle_field: ctypes.c_uint64, handle_grid: ctypes.c_uint64):
#         res = self.lib.field_new(handle_field, handle_grid)
#         if res != 0:
#             raise Exception('Failed to initialize field')
#
#     def field_delete(self, handle_field: ctypes.c_uint64):
#         res = self.lib.grid_delete(handle_field)
#         if res != 0:
#             raise Exception('Failed to initialize grid')
#


def _add_header(path):
    include_directive = f"#include \"{path}\"\n"
    # add this header for all native modules
    wp.codegen.cpu_module_header += include_directive
    wp.codegen.cuda_module_header += include_directive

def _register_base_headers():
    include_path = os.path.abspath(os.path.dirname(__file__))
    _add_header(f"{include_path}/Index_3d.h")
    _add_header(f"{include_path}/dDataView.h")
    _add_header(f"{include_path}/ngh_idx.h")


def _register_dense_headers():
    include_path = os.path.abspath(os.path.dirname(__file__))
    _add_header(f"{include_path}/dense/dSpan.h")
    _add_header(f"{include_path}/dense/dPartition.h")
    _add_header(f"{include_path}/dense/dIndex.h")

def _register_base_builtins():
    #from wpne import index_3d, data_view, ngh_idx

    Index_3d.warp_register_builtins()
    DataView.warp_register_builtins()
    Ngh_idx.warp_register_builtins()


def _register_dense_builtins():
    from .dense import dIndex, dSpan, dPartition

    dIndex.register_builtins()
    dSpan.register_builtins()
    dPartition.register_builtins()


def init():
    # Get the path of the current script
    script_path = __file__

    # Get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(script_path))

    print(f"Directory containing the script: {script_dir}")

    wp.build.set_cpp_standard("c++17")
    wp.build.add_include_directory(script_dir)
    wp.build.add_preprocessor_macro_definition('NEON_WARP_COMPILATION')

    # It's a good idea to always clear the kernel cache when developing new native or codegen features
    wp.build.clear_kernel_cache()

    _register_base_headers()
    _register_dense_headers()

    _register_base_builtins()
    _register_dense_builtins()