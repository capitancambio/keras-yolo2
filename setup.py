#!/usr/bin/env python
# -*- coding: utf-8 -*

import os

from setuptools import find_packages, setup

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()
    install_requires = [i for i in install_requires if '://' not in i]

setup(
    name='kerolo2',
    version='0.0.1',
    packages=find_packages(exclude=('tests',)),
    include_package_data=True,
    zip_safe=False,
    description='Yolo2 implementation in keras',
    author='Javier Asensio-Cubero '
           '(based on https://github.com/experiencor/keras-yolo2)',
    author_email='capitan.cambio@gmail.com',
    license='MIT',
    long_description='https://github.com/capitancambio/keras-yolo2',
    install_requires=install_requires,
    entry_points={
        'console_scripts': ['gen-anchors=kerolo2.gen_anchors:main',
                            'train=kerolo2.train:main',
                            'predict=kerolo2.predict:main'],
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)
