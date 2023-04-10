FROM jupyter/minimal-notebook:python-3.10.10

USER root

COPY . ${HOME}

WORKDIR ${HOME}

RUN pip install . && \
    chown -R jovyan ${HOME}

USER jovyan
