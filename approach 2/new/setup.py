from setuptools import setup 

setup(
      name='adidom_custom_code',
      version='0.1',
      install_requires=['opencv-python', 'pybase64'],
      scripts=['predictor.py', 'preprocess.py'])