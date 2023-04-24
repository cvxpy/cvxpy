FROM jupyter/scipy-notebook:python-3.10.10

USER root

COPY . ${HOME}

WORKDIR ${HOME}

RUN pip install --no-cache-dir . && \
    chown -R jovyan ${HOME}

USER jovyan
