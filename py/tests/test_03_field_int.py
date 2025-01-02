from env_setup import update_pythonpath

update_pythonpath()

import os

import warp as wp
import neon



def test_field_int():
    # Get the path of the current script
    script_path = __file__
    # Get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(script_path))

    def conainer_kernel_generator(field):
        partition = field.get_partition(neon.Execution.device(), 0, neon.DataView.standard())
        @wp.func
        def user_foo(idx: neon.dense.dIndex):
            wp.neon_print(idx)
            value = wp.neon_read(partition, idx, 0)
            print(value)

        @wp.kernel
        def neon_kernel_test(span: neon.dense.dSpan):
            is_valid = wp.bool(True)
            myIdx = wp.neon_set(span, is_valid)
            if is_valid:
                user_foo(myIdx)

        return neon_kernel_test

    wp.config.mode = "debug"
    wp.config.llvm_cuda = False
    wp.config.verbose = True
    wp.verbose_warnings = True

    wp.init()
    neon.init()
    dev_idx = 0
    with wp.ScopedDevice(f"cuda:{dev_idx}"):
        bk = neon.Backend(runtime=neon.Backend.Runtime.stream,
                        dev_idx_list=[dev_idx])

        grid = neon.dense.dGrid(bk, neon.Index_3d(10, 10, 10))
        span_device_id0_standard = grid.get_span(neon.Execution.device(),
                                                 0,
                                                 neon.DataView.standard())
        print(span_device_id0_standard)

        field = grid.new_field(cardinality=1, dtype=wp.int32)

        container = conainer_kernel_generator(field)
        wp.launch(container, dim=1, inputs=[span_device_id0_standard])

    wp.synchronize()


if __name__ == "__main__":
    test_field_int()
