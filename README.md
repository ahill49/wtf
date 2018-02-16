# wtf

Getting started
——————————————-

1. Download and install anaconda Python 2.7 version
https://www.anaconda.com/download/#macos

2. Install tensorflow using conda
https://www.tensorflow.org/install/install_mac
>> conda create -n tensorflow python=2.7
>> source activate tensorflow
>> pip install --ignore-installed --upgrade \
 https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.5.0-py2-none-any.whl
>> source deactivate (to exit environment)

3. In the tensorflow environment, install scipy and scikit-learn
>> source activate tensorflow
>> conda install -c anaconda scipy
>> conda install -c anaconda scikit-learn

4. If you use the spyder editor, set 
Preferences -> Python interpreter variable to
/anaconda2/envs/tensorflow/bin/python

5. Download MapSwipe annotations from http://mapswipe.geog.uni-heidelberg.de/?id=2020

6. Unless provided in this repository, create api_key.txt file containing Bing maps API key. See https://msdn.microsoft.com/en-us/library/ff428642.aspx

7. Download images using ./lib/get_bing.py

