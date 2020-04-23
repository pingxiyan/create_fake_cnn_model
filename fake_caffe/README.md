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
    $ mkdir build
    $ cd build
    $ cmake -DUSE_OPENCV=OFF -DCPU_ONLY=ON ..
    $ make -j
    $ mkdir intall
    $ make install PREFIX=`pwd`/install
    $ export PYTHONPATH=`pwd`/install/python:${PYTHONPATH}

#### Creat fake model.
    Refer caffe layer doc: http://caffe.berkeleyvision.org/tutorial/layers.html
    
    Python virtualenv is recommanded.
    $ pip install virtualenv
	# Python 2:
	$ virtualenv env
	# Python 3
	$ python3 -m venv env

	# Enter virtual env
	$ source env/bin/activate 

	# Eixt virtual env
	$ deactivate

    Some dependencies:
    $ pip2 install protobuf
    $ pip2 install scikit-image

    Copy your expect data to current path: expected_result_sim.dat
    $ python2 ./create_fake_caffe_model.py

    Convert to IR
    $ pip3 install numpy
    $ pip3 install networkx
    $ pip3 install defusedxml

    $ cvt_to_ir.sh

    Q&A:
    If you still have problem. Maybe you must uninstall some conflict python lib.

#### Verify model.
    Internal script, how to test tiny yolo v2. <br>
    ssh://git@gitlab.devtools.intel.com:29418/xipingya/verify_nn.git <br>
    
