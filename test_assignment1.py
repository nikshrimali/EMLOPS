import pytest
import random
import glob
import os


import pandas as pd

df = pd.read_csv('metrics.csv')


README_CONTENT_CHECK_FOR = []


def test_readme_exists():
    '''Checks if README.md exists'''
    assert os.path.isfile("README.md", ), "README.md file missing!"


def test_readme_contents():
    '''Contents of readme file has been properly written or not'''
    readme = open("README.md", "r", encoding="utf-8")
    readme_words = readme.read().split()
    readme.close()
    assert len(readme_words) >= 50, "Make your README.md file interesting! Add atleast 100 words"


def test_readme_proper_description():
    '''Checks for the functions implemented has proper description or not'''
    READMELOOKSGOOD = True
    f = open("README.md", "r")
    content = f.read()
    f.close()
    for c in README_CONTENT_CHECK_FOR:
        if c not in content:
            READMELOOKSGOOD = False
            pass
    assert READMELOOKSGOOD == True, "You have not described all the functions/class well in your README.md file"


def test_readme_file_for_formatting():
    '''Checks for Readme File formatting'''
    f = open("README.md", "r")
    content = f.read()
    f.close()
    assert content.count("#") >= 5


def test_check_accuracy():
    assert int(df.tail(1)['Train_acc'].item()) > 70, "This model is unworthy, overall accuracy less than 70"


def test_check_class_wise_accuracy():
    assert int(df.tail(1)['Cats_Accuracy'].item()) > 70 and int(df.tail(1)['Dogs_Accuracy'].item()) > 70, "This model is unworthy class wise accuracy less than 70"

def test_check_pt():
    root_dir = os.getcwd()
    assert len(list(glob.glob('*.pt'))) == 0, "No model Uploads on GIT"


def test_check_zip():
    root_dir = os.getcwd()
    assert len(list(glob.glob('*.zip'))) == 0, "No model Uploads on GIT"