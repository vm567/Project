---
title: Sentiment Analysis App
emoji: 😻
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: 1.17.0
app_file: app.py
pinned: false
---






# Project

What is hugging face space ?
It is a platform to deploy streamlit or gradio based application

things that will be using in this project - 
github.com tpo host the code base
hugging stream face - to host the web application
github.dev - to edit the code base


process - write code - push to github - then the codebase to be deployed to hugging face

Code 
- created a virtual env to install necessary libraries ( in vscode use command ctrl+shift+p  and enter python create environment) vscode automatically creates a virtual environment for us
- install necessary libraries 
- write all the libraries in a file (cmnd  pip  freeze > requirements.txt)
- create a file name app.py
- push the code to your feature branch 
- follow this link for more reference https://huggingface.co/docs/hub/spaces-github-actions
- URL to the deployed application----- https://huggingface.co/spaces/vm567/sentiment-analysis-app

# Docker project

## Docker installation
1. Firstly, Install dockers application(latest) from google.

2. ![ss 1](https://user-images.githubusercontent.com/123666927/227820729-7832d5d6-2d54-41e4-b34f-828c60b0be8c.png)
Go to settings, select general and check for the WSL which was selected by default in my system
3. Update the WSL to the latest version. Reference: "https://code.visualstudio.com/docs/remote/wsl-tutorial" and " https://docs.docker.com/desktop/windows/wsl/"
![ss 2](https://user-images.githubusercontent.com/123666927/227820942-0d349f03-b1db-4fcc-b4fe-b39725a3f6b7.png)

4. Now, install Ubuntu
5. Open the Visual Studio code and download the WSL extension pack and then install remote control development extension pack which sets up a development environment for the system. Click ctrl+shift+p and connect to "WSL : connect to the default distro"
6. Now, open terminal and write "code ."![ss3](https://user-images.githubusercontent.com/123666927/227820667-0c349953-7ff9-49bf-b70b-b603eb58b98a.png)

Hugging Face URL for USPTO : https://huggingface.co/spaces/vm567/Finetuning_HUPD_dataset
