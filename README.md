### Build docker

    docker build -t ml_projects . 
    
### RUN docker with jupyter notebook

    docker run  -v $PWD:/code -p 8888:8888 --name=ml_projects -it ml_projects

### If required add library
    pipdeptree -fl > requirements.txt
