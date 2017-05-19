# vips_code

Hi, thanks for your interests in our VIPS paper. You can find this paper here: https://arxiv.org/pdf/1611.00340.pdf

For LDA experiments, I used python code from https://github.com/blei-lab/onlineldavb, and added a few lines to add noise to the expected sufficient statistics.
First, download wikipedia documents using download_wikipage.py. Then, run massivesimul_static.py, which calls and runs onlinewikipedia_static.py. Make sure you set the Data_PATH correctly once you download the wikipage data.  

For BLR experiments, I wrote code in python. The stroke dataset isn't publically available. If you want to get access to the dataset, please contact IMEDS (http://imeds.reaganudall.org/ResearchLab). Here, I put an example code with simulated data, called "test_stochastic_ours.py". 

For SBN experiments, I used matlab code from https://github.com/zhegan27/dsbn_aistats2015, and added a few lines to add noise to the expected sufficient statistics. Due to the size of the dataset, I didn't include the dataset here, but you can download it from https://drive.google.com/drive/u/0/folders/0B1HR6m3IZSO_SW1jS1ZtRXlpakU.
Once you download the dataset and put it in your working directory, run mnist_sbn_private.m to test our VIPS code for SBN. 

If you find any errors, concerns, questions, or ways to improve, please let me know.

Have a wonderful day!

May 19, 2017 @ Amsterdam, Netherlands

