FROM pytorch/pytorch
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
RUN pip install -r requirements.txt
