FROM ubuntu:22.04

# Install Java, Spark, Python, CUDA toolkit
RUN apt-get update && apt-get install -y \
    openjdk-11-jdk python3 python3-pip wget curl gnupg2 && \
    rm -rf /var/lib/apt/lists/*

# Install Spark
RUN wget https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz && \
    tar -xzf spark-3.5.1-bin-hadoop3.tgz -C /opt && \
    ln -s /opt/spark-3.5.1-bin-hadoop3 /opt/spark

# Install RAPIDS jars (example for Spark 3.5 + Scala 2.12)
RUN wget https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/25.08.0/rapids-4-spark_2.12-25.08.0.jar -P /opt/spark/jars &&\
    wget https://repo1.maven.org/maven2/ml/dmlc/xgboost4j-spark-gpu_2.12/3.0.4/xgboost4j-spark-gpu_2.12-3.0.4.jar -P /opt/spark/jars

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# Update PATH
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda init bash

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main &&\
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -n data-mining-spark -y python==3.11.9

# Verify conda install
RUN conda --version

ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH
