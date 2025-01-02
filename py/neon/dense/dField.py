import ctypes

import neon
import warp as wp


# from .dPartition import dPartitionInt as dPartitionInt


class dField(object):
    def __init__(self,
                 neon_gate: neon.Gate,
                 grid_handle: ctypes.c_void_p,
                 cardinality: ctypes.c_int,
                 dtype,
                 py_grid,
                 ):

        self.dtype = dtype
        if grid_handle == 0:
            raise Exception('DField: Invalid handle')

        self.neon_gate:neon.Gate =  neon_gate
        self.handle_type = ctypes.c_void_p
        self.handle: ctypes.c_uint64 = ctypes.c_void_p(0)
        self.grid_handle = grid_handle
        self.cardinality = cardinality
        self.py_grid = py_grid
        self.field_type = None
        self._set_field_type()
        self._help_load_api()
        self._help_field_new()

    def __del__(self):
        self.help_delete()
        pass

    def _set_field_type(self):
        def _set_via_suffix(suffix):
            s_native, s_wp, s_cytpes = suffix
            is_native = self.dtype == s_native
            is_ctypes = s_wp == self.dtype
            is_warp = self.dtype == s_cytpes
            if (is_native or is_warp or is_ctypes):
                self.suffix = f'_{s_native}'
                self.field_type = s_cytpes
                self.Partition_type = getattr(neon.dense.dPartition, f'dPartition{self.suffix}')
                return True
            return False

        supported_suffixes = [('bool', wp.bool, ctypes.c_bool),
                              ('int8', wp.int8, ctypes.c_int8),
                                ('uint8', wp.uint8, ctypes.c_uint8),
                                ('int32', wp.int32, ctypes.c_int32),
                                ('uint32', wp.uint32, ctypes.c_uint32),
                                ('int64', wp.int64, ctypes.c_int64),
                                ('uint64', wp.uint64, ctypes.c_uint64),
                                ('float32', wp.float32, ctypes.c_float),
                                ('float64', wp.float64, ctypes.c_double)]
        match_found = False
        for suffix in supported_suffixes:

            match_found = _set_via_suffix(suffix)
            if match_found:
                break
        if not match_found:
            raise Exception(f'dField: Unsupported data type ({self.dtype})')

    def _help_load_api(self):
        # Importing new functions
        ## new_field
        lib_obj = self.neon_gate.lib

        # ---------------------------------------------------------------------
        self.api_new = getattr(lib_obj, f'dGrid_dField_new{self.suffix}')
        self.api_new.argtypes = [ctypes.POINTER(self.handle_type),
                                 self.handle_type,
                                 ctypes.c_int]
        self.api_new.restype = ctypes.c_int

        # ---------------------------------------------------------------------
        self.api_delete = getattr(lib_obj, f'dGrid_dField_delete{self.suffix}')
        self.api_delete.argtypes = [ctypes.POINTER(self.handle_type)]
        self.api_delete.restype = ctypes.c_int

        ## get_partition
        self.api_get_partition = getattr(lib_obj, f'dGrid_dField_get_partition{self.suffix}')
        self.api_get_partition.argtypes = [
            self.handle_type,
            ctypes.POINTER(self.Partition_type),  # the span object
            neon.Execution,  # the execution type
            ctypes.c_int,  # the device id
            neon.DataView,  # the data view
        ]
        self.api_get_partition.restype = ctypes.c_int

        # # size partition
        # self.neon.lib.dGrid_dField_partition_size.argtypes = [
        #     ctypes.POINTER(self.Partition_type)]
        # self.neon.lib.dGrid_dField_partition_size.restype = ctypes.c_int

        # field read
        self.api_read = getattr(lib_obj, f'dGrid_dField_read{self.suffix}')
        self.api_read.argtypes = [self.handle_type,
                                  ctypes.POINTER(neon.Index_3d),
                                  ctypes.c_int]
        self.api_read.restype = self.field_type

        # field write
        self.api_write = getattr(lib_obj, f'dGrid_dField_write{self.suffix}')
        self.api_write.argtypes = [self.handle_type,
                                   ctypes.POINTER(neon.Index_3d),
                                   ctypes.c_int,
                                   self.field_type]
        self.api_write.restype = ctypes.c_int

        # field update host data
        self.api_update_host = getattr(lib_obj, f'dGrid_dField_update_host_data{self.suffix}')
        self.api_update_host.argtypes = [self.handle_type,
                                         ctypes.c_int]
        self.api_update_host.restype = ctypes.c_int

        # field update device data
        self.api_update_device = getattr(lib_obj, f'dGrid_dField_update_device_data{self.suffix}')
        self.api_update_device.argtypes = [self.handle_type,
                                           ctypes.c_int]
        self.api_update_device.restype = ctypes.c_int

        # export vti
        self.api_export_vti = getattr(lib_obj, f'dGrid_dField_to_vti{self.suffix}')
        self.api_export_vti.argtypes = [self.handle_type,
                                           ctypes.c_char_p,
                                           ctypes.c_char_p]
        self.api_export_vti.restype = ctypes.c_int

    def _help_field_new(self):
        if self.handle == 0:
            raise Exception('dGrid: Invalid handle')

        res = self.api_new(ctypes.pointer(self.handle),
                           self.grid_handle,
                           self.cardinality)
        if res != 0:
            raise Exception('dGrid: Failed to initialize field')

    def help_delete(self):
        if self.handle == 0:
            return
        res = self.api_delete(ctypes.pointer(self.handle))
        if res != 0:
            raise Exception('Failed to delete field')

    def get_grid(self):
        return self.py_grid

    def get_partition(self,
                      execution: neon.Execution,
                      c: ctypes.c_int,
                      data_view: neon.DataView
                      ):
        if self.handle == 0:
            raise Exception('dField: Invalid handle')

        partition = self.Partition_type()

        res = self.api_get_partition(self.handle,
                                     partition,
                                     execution,
                                     c,
                                     data_view)
        if res != 0:
            raise Exception('Failed to get partition')

        # ccp_size = self.neon.lib.dGrid_dField_partition_size(partition)
        # ctypes_size = ctypes.sizeof(partition)
        # 
        # if ccp_size != ctypes_size:
        #     raise Exception(f'Failed to get span: cpp_size {ccp_size} != ctypes_size {ctypes_size}')
        # 
        # # print(f"Partition {partition}")
        return partition

    def read(self, idx: neon.Index_3d, cardinality: ctypes.c_int):
        return self.api_read(self.handle,
                             idx,
                             cardinality)

    def write(self, idx: neon.Index_3d, cardinality: ctypes.c_int, newValue):
        return self.api_write(self.handle,
                              idx,
                              cardinality,
                              self.field_type(newValue))

    def update_host(self, streamSetId: ctypes.c_int):
        return self.api_update_host(self.handle,
                                    streamSetId)

    def update_device(self, streamSetId: ctypes.c_int):
        return self.api_update_device(self.handle,
                                      streamSetId)

    def export_vti(self, filename: str,
                   field_name: str = "field"):
        self.api_export_vti(self.handle, filename.encode('utf-8'), field_name.encode('utf-8'))

    @property
    def type(self):
        return self.dtype
