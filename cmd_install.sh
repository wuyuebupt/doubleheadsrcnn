export PYTHONPATH=$PWD/maskrcnn_pythonpath

mkdir maskrcnn_pythonpath/

python setup.py build develop  --install-dir maskrcnn_pythonpath/
