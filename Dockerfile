FROM tensorflow/tensorflow:latest-gpu

MAINTAINER Thomas Wood <thomas@synpon.com>

# Setup scene-graph in /app, the location we are putting code from the repo.
COPY ./scene-graph-TF-release /app/scene-graph-TF-release

# Move the data into the docker (This needs to be scripted)
COPY ./sg_checkpoint.zip /app/scene-graph-TF-release/sg_checkpoint.zip

# TODO: REMOVE THIS PART WE WILL NOT NEED IT. This is temporary
RUN mkdir /app/scene-graph-TF-release/data
COPY ./vg_data.zip /app/scene-graph-TF-release/data/vg_data.zip

# Setup the weights where we want them to be.
RUN cd /app/scene-graph-TF-release && \
    unzip sg_checkpoint.zip && \
    rm sg_checkpoint.zip && \
    cp checkpoints/dual_graph_vrd_final_iter2.ckpt.index checkpoints/dual_graph_vrd_final_iter2.ckpt && \
    cd data && \
    unzip vg_data.zip && \
    rm vg_data.zip && \
    cd /

# Install needed modules with pip
RUN pip install Cython easydict graphviz pyyaml

# Make the roi pooling layer library.
RUN cd /app/scene-graph-TF-release/lib && \
    make && \
    cp roi_pooling_layer/src/* /usr/local/lib/python2.7/dist-packages/tensorflow/user_ops && \
    cd /usr/local/lib/python2.7/dist-packages/tensorflow/user_ops && \
    TF_INC=/usr/local/lib/python2.7/dist-packages/tensorflow/include && \
    TF_LIB=/usr/local/lib/python2.7/dist-packages/tensorflow && \
    nvcc -std=c++11 -c -o roi_pooling_op_gpu.cu.o roi_pooling_op_gpu.cu.cc \
      -I $TF_INC -I /usr/local -L$TF_LIB -ltensorflow_framework \
      -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D_GLIBCXX_USE_CXX11_ABI=0  && \
    g++ -std=c++11 -shared -o roi_pooling_op_gpu.so roi_pooling_op.cc roi_pooling_op_gpu.cu.o \
      -I $TF_INC -fPIC -L /usr/local/cuda-9.2/lib64/ -L /usr/local/cuda-9.0/targets/x86_64-linux/lib \
      -lcudart -L$TF_LIB -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0 && \
    cp roi_pooling_op_gpu.so /app/scene-graph-TF-release/lib/roi_pooling_layer/roi_pooling_op_gpu.so


# TODO: MOVE THIS CODE TO A SCRIPT!
# RUN cd /app/scene-graph-TF-release && \
#    wget https://www.dropbox.com/s/2rgq9vcx1jpeyjp/sg_checkpoint.zip && \
#    unzip sg_checkpoint.zip && \
#    rm sg_checkpoint.zip
RUN cd /app/Faster-RCNN_TF && \
    wget https://www.dropbox.com/s/cfz3blmtmwj6bdh/VGGnet_fast_rcnn_iter_70000.ckpt



# We dont technically need tkinter but w/e. Dont fight make a right.
RUN apt-get update && \
    apt-get install -y python-tk


# Okay maybe I will do this differently later, but I am just going to start
# throwing it all in there now.

# Setup Faster-RCNN_TF in /app
COPY ./Faster-RCNN_TF /app/Faster-RCNN_TF

# Build the fork where 'bash make.sh' is commented out and copy the roi_pooling_op library over.
RUN cd /app/Faster-RCNN_TF/lib && \
    make && \
    cp /app/scene-graph-TF-release/lib/roi_pooling_layer/roi_pooling_op_gpu.so /app/Faster-RCNN_TF/lib/roi_pooling_layer/roi_pooling.so



# Setup working directory and standard entry command.
WORKDIR "/app"
CMD ["/bin/bash"]
