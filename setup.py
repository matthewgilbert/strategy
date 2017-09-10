# -*- coding: utf-8 -*-

from setuptools import setup
import re

# https://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package#7071358
VERSIONFILE = "strategy/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." %
                       (VERSIONFILE,))

#http://stackoverflow.com/questions/10718767/have-the-same-readme-both-in-markdown-and-restructuredtext#23265673
try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

setup(name='strategy',
      version=verstr,
      description='Trading strategies with python',
      long_description=read_md('README.md'),
      url='https://github.com/MatthewGilbert/strategy',
      author='Matthew Gilbert',
      author_email='matthew.gilbert12@gmail.com',
      license='MIT',
      platforms='any',
      install_requires=[
          'pandas>=0.18.0',
          'numpy',
          'mapping',
          'pandas_market_calendars>=0.8'
      ],
      packages=['strategy'],
      zip_safe=False)
