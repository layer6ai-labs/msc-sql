FROM nvcr.io/nvidia/pytorch:24.05-py3
# requirements
RUN pip install flash-attn --no-build-isolation
COPY requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt

ENV TRANSFORMERS_CACHE=/home/.cache/huggingface/hub

COPY . /msc_sql
WORKDIR /msc_sql


