# Base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy the environment.yml file to the container
COPY environment.yml .
COPY . .
RUN conda update -n base -c defaults conda
# Create a Conda environment and activate it
RUN conda env create -f environment.yml && \
    echo "conda activate pt" >> ~/.bashrc
SHELL ["conda", "run", "-n", "pt", "/bin/bash", "-c"]

# Expose the Streamlit default port
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set the default command to run Streamlit when the container starts
CMD ["streamlit", "run", "streamlit-meter_reader.py", "--server.port=8501", "--server.address=0.0.0.0"]