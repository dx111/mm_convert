#!/bin/bash
set -e

if [ "$1" == "ins" ]; then
	rm -rf build/
	rm -rf dist/
	rm -rf mm_convert.egg-info/
	python setup.py bdist_wheel 
	pip install dist/mm_convert-0.0.1-py3-none-any.whl  --force-reinstall
elif [ "$1" == "dev" ]; then
   python setup.py develop
fi