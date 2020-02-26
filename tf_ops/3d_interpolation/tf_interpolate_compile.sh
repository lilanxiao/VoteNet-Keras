# TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/lanxiaoli/anaconda3/envs/test/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda-9.2/include -I /home/lanxiaoli/anaconda3/envs/test/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.2/lib64/ -L/home/lanxiaoli/anaconda3/envs/test/lib/python3.6/site-packages/tensorflow -ltensorflow_framework -O2 
#-D_GLIBCXX_USE_CXX11_ABI=0

#/bin/bash
# CUDA_ROOT=/usr/local/cuda-9.2
# TF_ROOT=/home/lanxiaoli/anaconda3/envs/test/lib/python3.6/site-packages/tensorflow

# ${CUDA_ROOT}/bin/nvcc tf_interpolate.cpp -o tf_interpolate_so.so -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.2
#g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I ${TF_ROOT}/include -I ${CUDA_ROOT}/include -lcudart -L ${CUDA_ROOT}/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4 or above
# g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I ${TF_ROOT}/include -I ${CUDA_ROOT}/include -I ${TF_ROOT}/include/external/nsync/public -lcudart -L ${CUDA_ROOT}/lib64/ -L ${TF_ROOT} -ltensorflow_framework -O2 
# -D_GLIBCXX_USE_CXX11_ABI=0

