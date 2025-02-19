FROM continuumio/miniconda3:23.10.0-1


RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

RUN conda install python=3.10
RUN conda install conda-libmamba-solver
RUN conda install conda-forge::openslide=4.0.0
RUN conda config --set solver classic
RUN conda install conda-forge::gdal

WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

CMD ["uvicorn", "slide_provider:slide_endpoint", "--host", "0.0.0.0", "--port", "3306", "--reload"]
