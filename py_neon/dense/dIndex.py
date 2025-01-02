import ctypes
import typing
import py_neon.index_3d

class dIndex(ctypes.Structure):
    _fields_ = [("location", py_neon.Index_3d)]

    def __init__(self,
                 val: py_neon.index_3d.Index_3d):
        self.location = val

    def __len__(self):
        return 3

    def __getitem__(self, index):
        if index == 0:
            return self.location.x
        if index == 1:
            return self.location.y
        if index == 2:
            return self.location.z
        raise IndexError("Index out of range")

    def __str__(self):
        str = f"({self.location.x}, "
        str += f"{self.location.y}, "
        str += f"{self.location.z})"
        str += "<Index_3d: addr=%ld>" % (ctypes.addressof(self))

        return str

    def __eq__(self, other):
        if not isinstance(other, dIndex):
            return NotImplemented
        return self.location == other.location
