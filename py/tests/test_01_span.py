from env_setup import update_pythonpath
update_pythonpath()

import os

import warp as wp

import neon
# from neon import Index_3d
# from neon.dense import dSpan


wp.config.mode = "debug"
wp.config.llvm_cuda = False
wp.config.verbose = True
wp.verbose_warnings = True

wp.init()
neon.init()

# import typing

@wp.func
def user_foo(idx: neon.dense.dIndex):
    wp.neon_print(idx)


@wp.kernel
def neon_kernel_test(span: neon.dense.dSpan):
    # this is a Warp array which wraps the image data
    is_valid = wp.bool(True)
    myIdx = wp.neon_set(span, is_valid)
    if is_valid:
        user_foo(myIdx)

def test_00_index3d():
    with wp.ScopedDevice("cuda:0"):
        bk = neon.Backend(runtime=neon.Backend.Runtime.stream,
                        n_dev=1)
        print("done")
        grid = neon.dense.dGrid(bk)
        span_device_id0_standard = grid.get_span(neon.Execution.device(),
                                                 0,
                                                 neon.DataView.standard())
        print(span_device_id0_standard)
        wp.launch(neon_kernel_test, dim=10, inputs=[span_device_id0_standard])
        wp.synchronize_device()



if __name__ == "__main__":
    test_00_index3d()