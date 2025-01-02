from env_setup import update_pythonpath
update_pythonpath()

import warp as wp
import os
import neon

def test_00_index3d():


    wp.config.mode = "debug"
    wp.config.llvm_cuda = False
    wp.config.verbose = True
    wp.verbose_warnings = True

    wp.init()
    neon.init()


    @wp.kernel
    def index_print_kernel(idx: neon.Index_3d):
        wp.neon_print(idx)


    @wp.kernel
    def index_create_kernel():
        idx = wp.neon_idx_3d(17, 42, 99)
        wp.neon_print(idx)


    with wp.ScopedDevice("cuda:0"):
        # pass index to a kernel
        idx = neon.Index_3d(11, 22, 33)
        wp.launch(index_print_kernel, dim=1, inputs=[idx])

        # create index in a kernel
        wp.launch(index_create_kernel, dim=1, inputs=[])

        wp.synchronize_device()

if __name__ == "__main__":
    test_00_index3d()