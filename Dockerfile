FROM python:3.6

RUN git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && \
    make -j4 && \
    cd python-package; python setup.py install

# setup jupyter with extensions
RUN pip install jupyter jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable codefolding/main
RUN jupyter nbextensions_configurator enable --user
RUN jupyter nbextension install https://github.com/kenkoooo/jupyter-autopep8/archive/master.zip --user
RUN jupyter nbextension enable jupyter-autopep8-master/jupyter-autopep8
RUN git clone git://github.com/moble/jupyter_boilerplate
RUN jupyter nbextension install jupyter_boilerplate
RUN jupyter nbextension enable jupyter_boilerplate/main
RUN jupyter nbextension enable --py widgetsnbextension --sys-prefix

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir /code
WORKDIR /code
COPY . /code

CMD jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root --notebook-dir="$PWD"
