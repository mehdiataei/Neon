import ctypes
import typing

import neon


# from neon import neon


class Report(object):
    def __init__(self, report_name: str = 'report'):

        self.handle: ctypes.c_void_p = ctypes.c_void_p(0)
        self.report_name = c_string = report_name.encode('utf-8')

        try:
            self.neon_gate: neon.Gate = neon.Gate()
        except Exception as e:
            self.handle = ctypes.c_void_p(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self.help_load_api()
        self.help_new()

    def __del__(self):
        if self.handle == 0:
            return
        self.help_delete()
        pass

    def help_load_api(self):
        self.api = {}

        def register_foo(foo_name, argtypes, restype):
            foo = getattr(self.neon_gate.lib, foo_name, None)
            # add a class variable with the name stored in the name variable
            foo.argtypes = argtypes
            foo.restype = restype
            self.api[foo_name] = foo
            pass

        # ------------------------------------------------------------------
        register_foo('report_new',
                     [ctypes.POINTER(self.neon_gate.handle_type),
                              ctypes.c_char_p],
                     ctypes.c_int)
        # ------------------------------------------------------------------
        register_foo('report_delete',
                     [ctypes.POINTER(self.neon_gate.handle_type)],
                     ctypes.c_int)
        # ------------------------------------------------------------------
        types = {
            'int64': ctypes.c_int64,
            'double': ctypes.c_double,
            'string': ctypes.c_char_p
        }
        for type_key in types.keys():
            register_foo( f'report_add_member_{type_key}',
                         [
                             self.neon_gate.handle_type,
                             ctypes.c_char_p,
                             types[type_key]],
                         ctypes.c_int)
            if  type_key == 'string':
                continue
            register_foo(f'report_add_member_vector_{type_key}',
                         [
                             self.neon_gate.handle_type,
                             ctypes.c_char_p,
                             ctypes.c_int,
                             ctypes.POINTER(types[type_key])],
                         ctypes.c_int)
        # ------------------------------------------------------------------
        register_foo('report_write',
                     [
                         self.neon_gate.handle_type,
                         ctypes.c_char_p,
                         ctypes.c_bool],
                     ctypes.c_int)

    def help_new(self):
        ret = self.neon_gate.lib.report_new(self.handle, self.report_name)
        if ret != 0:
            raise Exception("Error creating report")
            pass

    def help_delete(self):
        if self.handle == 0:
            return
        res = self.neon_gate.lib.report_delete(ctypes.pointer(self.handle))
        if res != 0:
            raise Exception('Failed to delete backend')

    def add_member(self, name: str, value: int):
        type_key = None
        if isinstance(value, int):
            type_key = 'int64'
        elif isinstance(value, float):
            type_key = 'double'
        elif isinstance(value, str):
            type_key = 'string'
        else:
            raise Exception('Invalid type')
        foo_name = f'report_add_member_{type_key}'
        if type_key == 'string':
            value = value.encode('utf-8')
        ret = self.api[foo_name](self.handle, name.encode('utf-8'), value)
        if ret != 0:
            raise Exception("Error adding member")
            pass

    def add_member_vector(self, name: str, value_vec: typing.List):
        import array
        def list_to_c_array(input_list):
            """
            Convert a list into a C-style array, managing the type dynamically.

            Parameters:
                input_list (list): The list to be converted.

            Returns:
                array.array: A C-style array containing the same values as the input list.

            Raises:
                ValueError: If the input list contains mixed or unsupported types.
            """
            if not isinstance(input_list, list):
                if isinstance(input_list, array):
                    return input_list
                raise ValueError("Input must be a list.")

            if not input_list:
                raise ValueError("Input list must not be empty.")

            # Infer the type of the list
            first_item = input_list[0]
            inferred_type = type(first_item)

            # Ensure all items in the list are of the same type
            if not all(isinstance(item, inferred_type) for item in input_list):
                raise ValueError("All items in the list must be of the same type.")

            # Map Python types to array module type codes
            # Supported types are double and int64
            type_mapping = {
                int: 'q',
                float: 'd'
            }

            if inferred_type not in type_mapping:
                raise ValueError(f"Unsupported data type: {inferred_type}")

            # Create a c-type array from the list to pass to a C function
            return array.array(type_mapping[inferred_type], input_list)

        array_view = list_to_c_array(value_vec)
        array_size = int(len(array_view))
        array_element = array_view[0]
        type_key = None
        if isinstance(array_element, int):
            type_key = 'int64'
            c_array_type = ctypes.c_int64 * array_size
        elif isinstance(array_element, float):
            type_key = 'double'
            c_array_type = ctypes.c_double * array_size
        else:
            raise Exception('Invalid type')

        c_array = c_array_type(*array_view)
        foo_name = f'report_add_member_vector_{type_key}'
        ret = self.api[foo_name](self.handle,
                             name.encode('utf-8'),
                             array_size,
                             c_array)
        if ret != 0:
            raise Exception("Error adding member")

    def write(self, name: str, value: bool=True):
        ret = self.api['report_write'](self.handle, name.encode('utf-8'), value)
        if ret != 0:
            raise Exception("Error writing report")
            pass


def try_report():
    report = Report()
    report.add_member('int_member', 10)
    report.add_member('float_member', 10.0)
    report.add_member_vector('int_vector', [1, 2, 3, 4, 5])
    report.add_member_vector('float_vector', [1.0, 2.0, 3.0, 4.0, 5.0])
    report.write('report', True)
    pass

if __name__ == '__main__':
    try_report()
    pass