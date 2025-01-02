import ctypes
import typing


class Ngh_idx(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int8),
                ("y", ctypes.c_int8),
                ("z", ctypes.c_int8)]

    def __init__(self,
                 x: int,
                 y: int,
                 z: int):
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return 3

    def __getitem__(self, index):
        if index == 0:
            return self.x
        if index == 1:
            return self.y
        if index == 2:
            return self.z
        raise IndexError("Index out of range")

    def to_wp_kernel_dim(self) -> typing.Tuple[int, int, int]:
        return (self.x, self.y, self.z)

    def __str__(self):
        str = "<Index_3d: addr=%ld>" % (ctypes.addressof(self))
        str += f"\n\tx: {self.x}"
        str += f"\n\ty: {self.y}"
        str += f"\n\tz: {self.z}"
        return str

    def __eq__(self, other):
        if not isinstance(other, Index_3d):
            return NotImplemented
        return (self.x == other.x and self.y == other.y and self.z == other.z)


    @staticmethod
    def warp_register_builtins():
        import warp as wp
        # register type
        wp.types.add_type(Ngh_idx, native_name="NeonNghIdx")

        # create dense index
        wp.context.add_builtin(
            "neon_ngh_idx",
            input_types={"x": wp.int8, "y": wp.int8, "z": wp.int8},
            value_type=Ngh_idx,
            missing_grad=True,
        )

        # create dense index
        wp.context.add_builtin(
            "neon_ngh_idx",
            input_types={"idx": Ngh_idx, "x": wp.int8, "y": wp.int8, "z": wp.int8},
            value_type=None,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_ngh_idx",
            input_types={"idx": Ngh_idx},
            value_type=wp.int8,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_get_y",
            input_types={"idx": Ngh_idx},
            value_type=wp.int8,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_get_z",
            input_types={"idx": Ngh_idx},
            value_type=wp.int8,
            missing_grad=True,
        )

        # print dense index
        wp.context.add_builtin(
            "neon_print",
            input_types={"a": Ngh_idx},
            value_type=None,
            missing_grad=True,
        )
