This zipped file contains all the source code to generate all results for Group 19's NLP Project, Fine-Tuning LMs with Conditioning on Two Human Preferences.


***************
RUNNING A DEMO
***************
The fine-tuned models were too large to upload to Moodle. Instead we have a link to a shared Google Drive folder which has all models, and a notebook to run a demo of evaluating our model on a very small evaluation set.

DEMO INSTRUCTIONS
*****************
Accessing the Demo Notebook:
1. open shared folder link: https://drive.google.com/drive/folders/1xsUmPY_1kRwpt1Hg_BkUKx1jM29nFfun?usp=sharing
2. Your Google Drive should have opened. Click "Shared with me" on the left menu
3. right-click on the shared "group_19_demo" folder -> click "Add a shortcut to drive" -> Select anywhere in your "MyDrive"
4. Open the "group_19_demo" folder recently added to your "MrDrive" -> open "demo_dataset_train_evaluate" colab notebook

Running the Demo Notebook:
5. You should now be on Colab -> Activate a GPU session (click Connect in top right -> RAM/Disk in the top right -> change runtime type -> activate GPU)
6. In the 2nd cell change the "%cd your_path_to_group_19_demo" to include your path to the folder, eg. "%cd /content/drive/MyDrive/group_19_demo"
7a. EITHER: Click "Run All" to create a dataset, fine-tuning a model, and evaluate on a very small dataset.
7b. OR: to only run the evaluation on a very small dataset; i) run the initial cells upto installing requirements, ii) then run cell under step 3 in notebook



*********************************************************
REPLICATING REPORT RESULTS FROM SCRATCH USING SOURCE CODE
*********************************************************
This zipped file contains all code required to:
1. Download gpt2 tokenizer
2. create train/valid dataset for fine-tuning
3. fine-tune models on a variety of conditional tokens
4. Evaluate all models using validation dataset

To do this run the following scripts on Google Colab:
1. !python scripts/tokenizer_create.py
2. !python main_dataset_create.py args/args_dataset_full
3. !python main_train args/args_train_tox
3. !python main_train args/args_train_sen
3. !python main_train args/args_train_toxsen
3. !python main_train args/args_train_none
4. !python main_evaluate.py args/args_evaluate_main1
5. !python main_evaluate.py args/args_evaluate_main2
6. !python main_evaluate.py args/args_evaluate_extra