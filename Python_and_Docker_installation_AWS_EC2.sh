sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8
alias python='python3.8'
python --version
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get update
sudo apt-get install     apt-transport-https     ca-certificates     curl     gnupg-agent     software$
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo docker ps
sudo docker images
git clone https://github.com/tensorflow/ecosystem.git
cd ecosystem/spark/spark-tensorflow-distributor/
curl -L https://github.com/docker/compose/releases/download/1.22.0/docker-compose-`uname -s`-`uname -m` > docker-compose
sudo chmod +x docker-compose
sudo mv docker-compose /usr/bin
sudo docker-compose build --build-arg PYTHON_INSTALL_VERSION=3.7
./tests/integration/run.sh

