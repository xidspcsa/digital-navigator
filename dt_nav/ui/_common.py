import os
import sys


def fix_import():
    git_directory = os.path.dirname(os.path.realpath(__file__))
    while (
        not os.path.exists(os.path.join(git_directory, ".git")) and git_directory != "/"
    ):
        git_directory = os.path.dirname(git_directory)
    if git_directory == "/":
        raise Exception("Could not find git directory")
    sys.path.append(git_directory)
