import os
import sys
import logging
logging.basicConfig(level=logging.DEBUG)

def update_pythonpath():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    sys.path.insert(0, script_dir+'/../')
    # print PYTHONPATH
    print(f"PYTHONPATH: {sys.path}")
    logging.debug(f"PYTHONPATH: {sys.path}")