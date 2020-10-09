import os

def location(rel_path):

     script_dir = os.path.dirname(__file__)
     abs_file_path = os.path.join(script_dir, rel_path)

     return abs_file_path