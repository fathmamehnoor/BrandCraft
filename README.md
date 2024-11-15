# BrandCraft

## Project Overview

This project is an AI-powered branding solution that generates taglines and logos for companies using GPT-2 and Stable Diffusion models. It helps businesses create their brand identity quickly and efficiently. The project is set up to run on **NVIDIA AI Workbench**, which provides an environment with pre-configured dependencies for easy access to the AI models and an intuitive UI for generating branding content.

## Description

BrandCraft leverages state-of-the-art machine learning models to generate creative taglines based on company names and descriptions. It also generates logos using diffusion models, providing a comprehensive branding experience. The platform allows users to input their company details and instantly receive slogans and logos that reflect their brand.

## Getting Started with BrandCraft on NVIDIA AI Workbench

1. Make sure you have NVIDIA AI Workbench installed. 

### Clone this repo with AI Workbench

1. Open NVIDIA AI Workbench.
2. Choose the location where you want to clone the project (e.g., **Local**).
3. If this is your first project, click the green **Clone Existing Project** button.  
   - Otherwise, click **Clone Project** in the top-right corner.
4. Paste in the repository URL, leave the default path, and click **Clone**.
![Clone Repo](/assets/image4.png)

### After cloning the project:
 
1. The container may take a few minutes to build; check the **Build Status** widget at the bottom of the AI Workbench window, expand it to view the output, and once it shows **Build Ready**, you’re good to proceed.

2. Install the model files and tokenizer configuration for loading the pre-trained model and upload into models/ [Download](https://drive.google.com/drive/folders/1aFP4VzJ0qT8Lkbvfh9rdotDU_rcuADjD?dmr=1&ec=wgc-drive-globalnav-goto)
![Models Folder](/assets/image3.png)

### Start the BrandCraft application

1. Ensure that the container has finished building.
2. When it’s ready, click the green **Open Application** button in the top-right to start the Gradio-based BrandCraft interface.
![Opening Gradio App](/assets/image2.png)



