# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set the working directory in the container to /app
WORKDIR /

# Copy the current directory contents into the container at /app
COPY ./backend ./backend
COPY ./requirements.txt ./requirements.txt
COPY ./config.yml ./config.yml

# Update the sources list
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip &&\
    pip install --no-cache-dir -r ./requirements.txt

RUN pip install git+https://github.com/facebookresearch/segment-anything.git

# Make port 8080 available to the world outside this container
EXPOSE 8080
EXPOSE 8080

# Define environment variable
ENV NAME few_sam

# Run app.py when the container launches
CMD ["python", "./backend/server.py"]
CMD ["python", "./backend/forwarder.py"]