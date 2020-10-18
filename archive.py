# -*- coding: utf-8 -*-
import os

if __name__ == '__main__':
    os.popen("git archive -o code_`git rev-parse HEAD | cut -c 1-5`.zip HEAD")
