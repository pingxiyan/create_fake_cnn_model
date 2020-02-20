# Guide.

#### Design your fake network
    Visualizor design tool.

#### Ready pycaffe. 

    $ sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
    $ sudo apt-get install python-dev
    $ sudo apt-get install libatlas-base-dev
    $ sudo apt-get install --no-install-recommends libboost-all-dev
    $ sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev

    $ git clone https://github.com/BVLC/caffe.git
    $ cd caffe

    Modify Makefile.config.example->Makefile.config
    $ diff Makefile.config.example Makefile.config
	8c8
	< # CPU_ONLY := 1
	---
	> CPU_ONLY := 1
	11c11
	< # USE_OPENCV := 0
	---
	> USE_OPENCV := 0
	15c15
	< # USE_HDF5 := 0
	---
	> USE_HDF5 := 0
	94c94
	< # WITH_PYTHON_LAYER := 1
	---
	> WITH_PYTHON_LAYER := 1


    $ make pycaffe
    $ export PYTHONPATH=`pwd`/python:${PYTHONPATH}

#### Creat fake model.
    Refer caffe layer doc: http://caffe.berkeleyvision.org/tutorial/layers.html

    $ python2 ./create_fake_caffe_model.py

#### Verify model.

