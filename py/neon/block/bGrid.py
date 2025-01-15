import ctypes

import neon
from .bField import bField
from neon.execution import Execution
from .bSpan import bSpan
from neon.index_3d import Index_3d
import numpy as np

class bGrid(object):
    def __init__(self, backend = None, dim = None, sparsity_pattern: np.ndarray = None):
        if sparsity_pattern is None:
            sparsity_pattern = np.ones((dim.x,dim.y,dim.z))
        if backend is None:
            # raise exception
            raise Exception('dGrid: backend pamrameter is missing')
        if sparsity_pattern.shape[0] != dim.x or sparsity_pattern.shape[1] != dim.y or sparsity_pattern.shape[2] != dim.z:
            raise Exception('dGrid: sparsity_pattern\'s shape does not match the dim')

        self.handle: ctypes.c_void_p = ctypes.c_void_p(0)
        self.backend = backend
        self.dim = dim
        self.sparsity_pattern = sparsity_pattern

        self._help_load_api()
        self._help_grid_new()

    def __del__(self):
        if self.handle == 0:
            return
        self._help_grid_delete()

    def _help_load_api(self):
        try:
            self.neon_gate: neon.Gate = neon.Gate()
        except Exception as e:
            self.grid_handle: ctypes.c_void_p = ctypes.c_void_p(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        # grid_new
        lib = self.neon_gate.lib
        self.api_new = lib.bGrid_new
        self.api_new.argtypes = [ctypes.POINTER(self.neon_gate.handle_type),
                                 self.neon_gate.handle_type,
                                 ctypes.POINTER(neon.Index_3d),
                                 ctypes.POINTER(ctypes.c_int),
                                 ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_int)]
        self.api_new.restype = ctypes.c_int

        # grid_delete
        self.api_delete = lib.bGrid_delete
        self.api_delete.argtypes = [ctypes.POINTER(self.neon_gate.handle_type)]
        self.api_delete.restype = ctypes.c_int
        # get_dimensions
        self.api_get_dimensions = lib.bGrid_get_dimensions
        self.api_get_dimensions.argtypes = [self.neon_gate.handle_type,
                                            ctypes.POINTER(neon.Index_3d)]
        self.api_get_dimensions.restype = ctypes.c_int

        # get_span
        self.api_get_span= lib.bGrid_get_span
        self.api_get_span.argtypes = [self.neon_gate.handle_type,
                                      ctypes.POINTER(bSpan),  # the span object
                                      neon.Execution,  # the execution type
                                      ctypes.c_int,  # the device id
                                      neon.DataView,  # the data view
                                      ]
        self.api_get_span.restype = ctypes.c_int

        # span_size
        self.api_span_size  = lib.bGrid_span_size
        self.api_span_size.argtypes = [ctypes.POINTER(bSpan)]
        self.api_span_size.restype = ctypes.c_int

        # get_properties
        self.api_get_properties=lib.bGrid_get_properties
        self.api_get_properties.argtypes = [self.neon_gate.handle_type,
                                            ctypes.POINTER(neon.Index_3d)]
        self.api_get_properties.restype = ctypes.c_int

        # is_inside_domain
        self.api_is_inside_domain=lib.bGrid_is_inside_domain
        self.api_is_inside_domain.argtypes = [self.neon_gate.handle_type,
                                              ctypes.POINTER(neon.Index_3d)]
        self.api_is_inside_domain.restype = ctypes.c_bool


    def _help_grid_new(self):
        if self.backend.handle.value == 0:  # Check backend handle validity
            raise Exception('bGrid: Invalid backend handle')

        if self.handle.value != 0:  # Ensure the grid handle is uninitialized
            raise Exception('bGrid: Grid handle already initialized')

        sparsity_pattern_array = self.sparsity_pattern.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        res = self.api_new(ctypes.pointer(self.handle),
                           self.backend.handle,
                           self.dim,
                           sparsity_pattern_array)
        if res != 0:
            raise Exception('bGrid: Failed to initialize grid')
        print(f"bGrid initialized with handle {self.handle.value}")

    def _help_grid_delete(self):
        if self.api_delete(ctypes.pointer(self.handle)) != 0:
            raise Exception('Failed to delete grid')

    def get_python_dimensions(self):
        return self.dim

    def get_cpp_dimensions(self):
        cpp_dim = Index_3d(0,0,0)
        res = self.api_get_dimensions(self.handle, cpp_dim)
        if res != 0:
            raise Exception('bGrid: Failed to obtain grid dimension')

        return cpp_dim

    def new_field(self, cardinality: ctypes.c_int) -> bField:
        field = bField(self.handle, cardinality)
        return field

    def get_span(self,
                 execution: Execution,
                 c: ctypes.c_int,
                 data_view: neon.DataView) -> bSpan:
        if self.handle == 0:
            raise Exception('bGrid: Invalid handle')

        span = bSpan()
        res = self.api_get_span(self.handle,
                                ctypes.addressof(span),
                                execution,
                                c,
                                data_view)
        if res != 0:
            raise Exception('Failed to get span')

        cpp_size = self.api_span_size(span)
        ctypes_size = ctypes.sizeof(span)

        if cpp_size != ctypes_size:
            raise Exception(f'Failed to get span: cpp_size {cpp_size} != ctypes_size {ctypes_size}')

        return span

    # def getProperties(self, idx: Index_3d):
    #     return DataView(self.neon_gate.lib.bGrid_get_properties(ctypes.byref(self.handle), idx))

    # for some reason, negative numbers in the index will return true for bGrids.
    def isInsideDomain(self, idx: Index_3d):
        if idx.x < 0 or idx.y < 0 or idx.z < 0:
            raise Exception('can\'t access negative indices in mGrid') # @TODOMATT make sure that this is a valid requirement
        return self.neon_gate.lib.bGrid_is_inside_domain(ctypes.byref(self.handle), idx)
