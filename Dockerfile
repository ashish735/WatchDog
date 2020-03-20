FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

LABEL ashish kumar

ARG username=pytorch_ashish
ARG userid=10301

COPY requirements.txt /requirements.txt

RUN apt-get clean \
        && apt-get update \
        && apt-get install -y ffmpeg libportaudio2 vnc4server vim wmctrl curl apt-transport-https libasound2 \
		libboost-all-dev build-essential libboost-thread-dev \
		apt-utils\
		bzip2 \
                zip \
                coreutils \
                unzip \
                nano \
		curl \
		g++ \
		git \
		graphviz \
		libgl1-mesa-glx \
		libhdf5-dev \
		openmpi-bin \
		wget

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

#install anaconda
RUN wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh

#RUN curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg; \
 #   install -o root -g root -m 644 microsoft.gpg /etc/apt/trusted.gpg.d/; \
 #   sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list'; \
 #   apt-get update; \
 #   apt-get install -y code; \
#   sed -i 's/BIG-REQUESTS/_IG-REQUESTS/' /usr/lib/x86_64-linux-gnu/libxcb.so.1

#RUN wget https://nodejs.org/dist/v10.16.0/node-v10.16.0.tar.gz; \
#    tar -xzvf node-v10.16.0.tar.gz; \
 #   cd node-v10.16.0; \
  #  ./configure; \
   # make; \
   # make install; \
    #rm ../node-v10.16.0.tar.gz;

RUN useradd -u ${userid} -ms /bin/bash -N ${username}
RUN chown ${username} $CONDA_DIR -R
COPY jupyter_notebook_config.py /
RUN chown ${username} jupyter_notebook_config.py
RUN mkdir /home/${username}/deep_learning
RUN chown ${username} /home/${username}/deep_learning

USER ${username}

#ARG python_version=3.6
RUN conda install -y python=3.6
RUN conda install -y tensorflow==2.1.0
#RUN conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#ENV PASSWORD=123456 WIDTH=1920 HEIGHT=1080
#COPY ./noVNC/ /noVNC/
#COPY ./startup.sh /startup.sh
#RUN chmod 777 /startup.sh; \
#    chmod 777 -R /noVNC


EXPOSE 8888 6006 5901

WORKDIR /home/${username}/deep_learning

#ENTRYPOINT ["/startup.sh"]

CMD jupyter lab --config=/jupyter_notebook_config.py --no-browser --port=8888 --ip=0.0.0.0
