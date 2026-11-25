from importlib.metadata import distribution


def pre_find_module_path(api):
    api.search_dirs = [str(distribution(api.module_name).locate_file(""))]
