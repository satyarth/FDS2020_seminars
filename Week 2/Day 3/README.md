# Quiz: Large Scale Software Engineering Data Science

Created: Apr 24, 2020 5:54 AM
Lesson Date: Apr 24, 2020
Status: Not Started
Type: Quiz

Your objective in this task is to put into practice the main tools that you have learned:

- Bash
- Git
- Design Patterns / Antipatterns
- Parallelisation and large scale Machine Learning with Dask

### Task Description

- For this, we will take as a start point the script to load the datasets discussed during the tutorial. It is available in our git repository as `main.py`
- You will have to fork the repository, in case you have not done it, so you can clone it in your local environment
- The script is relatively clean, but it still has various hard dependencies such as:
    - Number of rows to use
    - URL to download from
    - Location in the hard drive to store the data
- You will start by separating those dependencies and use the command line where necessary to pass arguments
- After this, you will be working with Dask Dataframes and Dask ML
- The objective is to predict the **delay time** for a flight.
    - We will explore two ways to do this, one is using a Scikit Regression Model of your choice
    - The second option will be using Dask ML directly
        - You might want to use the XBRegressor available in Dask ML

[dask_ml.xgboost.XGBRegressor - dask-ml 1.3.1 documentation](https://ml.dask.org/modules/generated/dask_ml.xgboost.XGBRegressor.html#dask_ml.xgboost.XGBRegressor)

- All this should follow the software engineering practices that we have discussed over the week such as modularity and separation of concerns
- If you feel inclined, you can do some feature engineering by computing aggregations such as mean delay, and so on. This is not required but it will help you become more familiar with Dask
- Remember to split your data set into train, test and validation
- You will use `dask.delayed` so we can use the full dataset and not be limited by the size of your RAM
- You have to use joblib for the gridsearch of hyperparameters.
    - Feel free to decide which parameters should be optimised for your two models.
    - Remember that many parameters lead to longer computing time, even in Dask
- You will then create a cluster and a client, so the Scikit regressor can be trained with the Dask backend
- After this, you will proceed to use the Dask ML regressor
- Store the accuracy of your models as well as the training times and the time required to do the grid search in a CSV file. Not only this is a good practice, but also **you will need it for the submission in Canvas!**
- Once your scripts are ready, please include all files in a folder with your name that you will include in the Submissions folder
- Use `.gitignore` to **NOT** commit the raw data files. Otherwise, Github will reject the commit. Here is an explanation

[.gitignore file - ignoring files in Git | Atlassian Git Tutorial](https://www.atlassian.com/git/tutorials/saving-changes/gitignore)

- Commit to your git repository and push it
- Create a pull request so it can be merged with the class repository
- You will have until **Tuesday the 28th of April 23:59 MSK** to complete this task plus 15 additional minutes to complete the quiz in Canvas. It will ask you the following:
    - (1x) The hash of your commit
    - (2x) The time required to compute the gridsearch
    - (2x) The time required to train each of the two models
    - (2x) The accuracy obtained in each of them
- Remember that the focus is not in getting the best possible accuracy, both in creating good software that can work with large datasets.

## Some notes

If you are using Google Colab, it does not seem to support Dask 100%, for example processes

```python
client = Client(processes=False)
```

Otherwise, Dask should run without problems in most laptops and local computers.

If you want to produce the visualizations, remember that you have to install `graphviz` both in conda as well as in your system. For Ubuntu, this would be:

```bash
sudo apt install python-pydot python-pydot-ng graphviz

conda install -c conda-forge graphviz

conda install -c conda-forge python-graphviz
```

If you are not using Linux, it can be more complicated to get it up and running, therefore, do not spend time on it and focus on the task. Dask does not require `graphviz` to run.