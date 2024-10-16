import pandas as pd
import random as rand   # For dummy scoring
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.sampling import BayesianModelSampling
import warnings
import logging
import copy

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# For tensorboard
import torch
from torch.utils.tensorboard import SummaryWriter

class MRK():

    def __init__(self, X = None, y = None, max_parents = 2, classifier='lr'):
        """
        self.X - the independent variables.
        self.y - the dependent variable.
        self.data - the entire dataset to be fit to.
        self.variable_names - list of names of all independent variables (X)
        self.target_col_name - the name of the dependent variable (y)
        self.model - the fitted model.
        """
        self.ordinal_encoder = OrdinalEncoder(dtype=int, handle_unknown='use_encoded_value', unknown_value=-1)
        self.label_encoder = LabelEncoder()
        self.writer = SummaryWriter() # Initialize the tensoroard writer
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_data = None
        self.test_data = None
        self.variable_names = None
        self.target_col_name = None
        self.model = None
        self.classifier = classifier
        self.top_score = None
        self.max_parents = max_parents
        self.max_vars_visited = 10

    def fit(self, X, y, test_size=0.5, verbose=False): 
        """
        X - the independent variables.
        y - the dependent variable.
        train_test_split_size - 
        Note: X and y must be supplied as pandas DataFrames or Series.
        Returns: self: (object) a fitted model ready for sampling.
        """
        # Input Validation: error handling for cases where X and y don't have the same number of instances, missing column headers etc..
        if not isinstance(X, (pd.DataFrame, pd.Series)): raise ValueError("X must be a pandas DataFrame or Series.")    # Validate that X and y are pandas DataFrames or Series
        if not isinstance(y, (pd.DataFrame, pd.Series)): raise ValueError("y must be a pandas DataFrame or Series.") 

        if len(X) != len(y): raise ValueError("X and y must have the same number of instances.")    # Validate that X and y have the same number of instances.

        self.variable_names = X.columns
        self.target_col_name = y.name

        # Encode X and y while retaining DataFrame/Series dtypes and column headers
        X = pd.DataFrame(self.ordinal_encoder.fit_transform(X), columns=self.variable_names)
        y = pd.Series(self.label_encoder.fit_transform(y).astype(int), name=self.target_col_name)

        # Train Test Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42) # try 50/50 split to make it less complex

        # Combine X y into train and test sets.
        self.train_data = pd.concat([self.X_train, self.y_train], axis=1)
        self.test_data = pd.concat([self.X_test, self.y_test], axis=1)

        # Step 1. Initialise Naive Bayes Structure.
        nb_structure = self.naive_bayes()   

        # Step 2. Determine an initial ordering for the X variables.
        ordered_variables = self.ordered_vars()     

        # Step 3. Learn the best structure via K2 with the initial ordering and the naive bayes structure.
        self.model, self.top_score = self.k2_structure_learning(ordered_variables, nb_structure)

        if verbose:
            print("Model has been fit.")
            print(f"TSTR Score of the optimal structure {self.top_score}")

        return self

    def naive_bayes(self): # Initialise a Bayesian Network with a Naive Bayes structure.
        """
        returns: A BayesianNetwork() object with a Naive Bayes structure.
        """
        model = BayesianNetwork()
        model.add_node(self.target_col_name)
        model.add_nodes_from(self.variable_names)
        
        for variable in self.variable_names:
            model.add_edge(self.target_col_name,variable)

        return model
    
    def ordered_vars(self, data=None, target=None):  # Returns an ordered list of variable names.

        if data is None: data = self.train_data  # no need to pass in the data itself, it can just be the column names.
        if target is None: target = self.target_col_name

        mutual_info_scores = self.mutual_information(data, target)  # Dict of scores
        sorted_var_names = sorted(mutual_info_scores, key=mutual_info_scores.get, reverse=True)  # List of dict keys in descending order of their values.

        return sorted_var_names

    def mutual_information(self, data, target, norm='none'):   # Mutual Information Scoring function.
        """
        data - A dataset as a pandas dataframe
        target - The target column name (y)
        norm - a keyword (either 'none', 'geometric_mean', etc..) to choose the type of normalisation
        Note: this scoring function is symmetric.
        returns: a dictionary where key=variable name and value = Mutual Information score.
        """
        X = data.columns.drop(target) # Drop the target column from the X columns
        var_score_dict = {col: 0 for col in X}  # Creates a dict where key='var name and value=score. Scores are initialised to 0.

        for var in var_score_dict.keys():   # Iterate through each independent variable
            mi_score = mutual_info_score(data[var], data[target])
            var_score_dict[var] = mi_score # Update the dict to reflect the new score
        
        # normalization_methods = {
        #     "none": lambda mi_score: mi_score,
        #     "geometric_mean": lambda mi_score, var_1_entropy, var_2_entropy: mi_score / np.sqrt(var_1_entropy * var_2_entropy),
        #     "arithmetic_mean": lambda mi_score, var_1_entropy, var_2_entropy: 2 * mi_score/ (var_1_entropy + var_2_entropy)}
        return var_score_dict
        
    def k2_structure_learning(self, ordered_var_names, naive_bayes_bn):    # Should accept the data and return the optimal structure as a Bayesian Network.
        """
        ordered_var_names - an ordered list of variable names.
        max_parents - the maximum number of parents allowed for any independent variable.
        naive_bayes_bn - a BayesianNetwork() object initialised with a Naive Bayes structure.
        """
        max_parents = self.max_parents
        max_vars_visited = self.max_vars_visited

        visited_vars = []
        best_candidate = copy.deepcopy(naive_bayes_bn) # Init the candidate as the Naive Bayes structure.
        best_candidate_score = self.meg_scoring_function([best_candidate], self.train_data)[0]
        step = 0

        for child in ordered_var_names:   # Iterate through the variables in order.
            parents = []
            ok_to_proceed = True
            while len(parents) < max_parents and len(visited_vars) < max_vars_visited and ok_to_proceed:  # Check if the max number of parents for the variable has been reached.

                candidate_structures = []   # Create empty arrays for candidate structures and candidate parents
                candidate_parents = []
                for parent in visited_vars: # Iterate through all the visited variables as candidate parents for the variable.
                    if parent not in parents:
                        new_candidate = copy.deepcopy(best_candidate)
                        new_candidate.add_edge(parent, child)
                        candidate_structures.append(new_candidate)  # Add the new structure to the candidate array.
                        candidate_parents.append(parent)

                if not candidate_structures:
                  break  # No candidates to consider

                else:
                  candidate_scores = self.meg_scoring_function(candidate_structures, self.train_data)
                  top_score_idx = candidate_scores.index(max(candidate_scores))   # Find the index of the max candidate score

                  if candidate_scores[top_score_idx] > best_candidate_score:
                      best_parent = candidate_parents[top_score_idx]
                      parents.append(best_parent)  # Add the new parent to the parents list
                      best_candidate = candidate_structures[top_score_idx]  # Update the best candidate structure
                      best_candidate_score = candidate_scores[top_score_idx]   # Update the best candidate score

                  else: ok_to_proceed = False  # No improvement, exit the loop

                  step += 1  # Increment the step counter
                  self.writer.add_scalar("TSTR Score", best_candidate_score, step)  # Log current score for the step

            visited_vars.append(child)  # Add the just visited child to the list of visited variables.

        return best_candidate, best_candidate_score
        
    def meg_scoring_function(self, array_of_structures, training_data, epochs=30, sample_size=None):  # Scores Bayesian Network structures against the real dataset.
        """
        array_of_structures - An array of Bayesian Network objects to be scored.
        training_data - the training dataset to which each structure will be fit.
        epochs - Number of epochs for evaluation (default 30).
        sample_size - how many instances to sample from each structure for evaluation
        Returns: An array of length n containing the respective score of each structure.
        """
        if sample_size == None: sample_size = len(self.X_train)
        array_of_scores = []  # Initialise empty array to house scores. 
        for structure in array_of_structures:  # For each BN in the array fit (fit with the pgmpy) to the training dataset (try 50/50 split to make it less complex)

            structure.remove_cpds(*structure.get_cpds())

            # incorporate meg here ->
            structure.fit(data=training_data, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10 )  # Fit the training data to the structure 
            sampler = BayesianModelSampling(structure)
            synthetic_data = sampler.forward_sample(sample_size)  # Generate synthetic data (can be pgmpy forward sampling method, or it can be GANBLR)
            ## <-

            # Split synthetic data into X and y
            synthetic_X = synthetic_data[self.variable_names] 
            synthetic_y = synthetic_data[self.target_col_name].astype(int)

            # Evaluate synthetic data with TSTR logreg for 30 epochs
            score = self.evaluate(synthetic_X, synthetic_y, self.X_test, self.y_test, epochs=epochs)
            array_of_scores.append(score)   # Populate the array of scores.

        return array_of_scores  # Returns an array of respective scores
    
    def dummy_scoring_function(self, array_of_structures, training_data, epochs=30, sample_size=100):
        """
        array_of_structures - An array of Bayesian Network objects to be scored.
        returns: An array of length n with respective scores.
        """
        n = len(array_of_structures) # n represents the number of structures to be scored.
        array_of_scores = []    # Initialise empty array to house scores. 
        for i in range(n):
            array_of_scores.append(rand.random())
        return array_of_scores  # Returns an array of respective scores
    
    def evaluate(self, X_synthetic, y_synthetic, X_real, y_real, epochs, model=None) -> float:
        """
        X_synthetic - The X data to be evaluated (X)
        y_synthetic - The y data to be evaluated (y_pred)
        X_real - Dependent variables from the real dataset (X_test)
        y_real - Target variable from the real dataset (y_test)
        """
        eval_model = None

        models = dict(
            lr=LogisticRegression,
            rf=RandomForestClassifier,
            mlp=MLPClassifier)
        if model == None:  model = self.classifier
        if model in models.keys():  eval_model = models[model]()
        else: raise Exception("Invalid Arugument `model`, Should be one of ['lr', 'mlp', 'rf'], or a model class that have sklearn-style `fit` and `predict` method.")

        total_accuracy = 0  # Cumulative accuracy score

        for epoch in range(epochs):
            # Set up and train the model pipeline
            pipeline = Pipeline([
                ('encoder', OneHotEncoder(categories='auto', handle_unknown='ignore')), 
                ('model',  eval_model)])

            pipeline.fit(X_synthetic, y_synthetic)
            pred = pipeline.predict(X_real) # Predict and return accuracy

            accuracy = accuracy_score(y_real, pred)
            total_accuracy += accuracy

        return total_accuracy / epochs
    
    def evaluate_trtr(self, X_train, y_train, X_test, y_test, epochs, model=None) -> float:
        """
        X_train - Dependent variables from the training set
        y_train - Target variable from the training set
        X_test - Dependent variables from the test set
        y_test - Target variable from the test set
        """
        if model == None:  model = self.classifier
        eval_model = None

        models = dict(
            lr=LogisticRegression,
            rf=RandomForestClassifier,
            mlp=MLPClassifier)
        
        if model in models.keys():  eval_model = models[model]()
        else: raise Exception("Invalid Arugument `model`, Should be one of ['lr', 'mlp', 'rf'], or a model class that have sklearn-style `fit` and `predict` method.")

        total_accuracy = 0  # Cumulative accuracy score

        for epoch in range(epochs):
            # Set up and train the model pipeline
            pipeline = Pipeline([
                ('encoder', OneHotEncoder(categories='auto', handle_unknown='ignore')), 
                ('model',  eval_model)])

            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test) # Predict and return accuracy

            accuracy = accuracy_score(y_test, pred)
            total_accuracy += accuracy

        return total_accuracy / epochs
    
    def results(self):
        '''
        Returns a df of results from the run. 
        '''
        trtr_accuracy = self.evaluate_trtr(self.X_train, self.y_train, self.X_test, self.y_test, epochs=10)

        results_df = pd.DataFrame(columns=['Classifier','TRTR','TSTR','Edges'])
        results_df.loc[len(results_df)] = [self.classifier, trtr_accuracy, self.top_score, self.model.edges()]

        return results_df
    

def main(args): # Main function which iterates through the datasets to test the model. 
    """
    Main function which iterates through the datasets to test the model. 
    Prints the results table to the terminal and to a .csv
    """
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('pgmpy').setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    def complete_runs(dataset_name, df, classifier, max_parents):  # Completes 10 runs and outputs the results as csv
        """
        Completes 10 runs of the MRK Algorithm
        Outputs the run results as a csv.
        Returns: Aggregated results as a dict. Incl: TRTR accuracy, TRTR Std Dev, TSTR accuracy, TSTR Std Dev
        """
        X_train = df.drop('class', axis=1)
        y_train = df['class']

        # Prepare dataframe to store the result of each run. 
        results_df = pd.DataFrame(columns=['Run', 'Time Completed', 'Classifier','TRTR','TSTR','Edges'])

        for i in range(1, 11):  # Complete 10 runs
            mrk = MRK(classifier=classifier, max_parents=max_parents)
            mrk.fit(X_train, y_train, verbose=False)
            run_result = mrk.results()
            run_result['Run'] = i
            run_result['Time Completed'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            results_df = pd.concat([results_df, run_result], ignore_index=True)

        # Output the results of each run to csv.
        results_df.to_csv(f"results/{dataset_name}_{datetime.now().strftime('%Y_%m_%d_%H%M')}.csv")
        
        # Aggregate the results
        trtr_mean = results_df['TRTR'].mean()
        trtr_std_dev = results_df['TRTR'].std()
        tstr_mean = results_df['TSTR'].mean()
        tstr_std_dev = results_df['TSTR'].std()
        # Compile Aggregated Results into a dataframe
        aggregate_dict = {
            'TRTR Mean': trtr_mean,
            'TRTR StdDev': trtr_std_dev,
            'TSTR Mean': tstr_mean,
            'TSTR StdDev':tstr_std_dev}

        return aggregate_dict

    def get_results_table(datasets=['adult'],classifiers=['lr'], max_parents=[2], istest=False):

        dataset_dict = {'adult': 'https://raw.githubusercontent.com/chriszhangpodo/discretizedata/main/adult-dm.csv',  # Discretized version of the adult dataset
                    'magic': 'https://raw.githubusercontent.com/chriszhangpodo/discretizedata/main/magic.csv',
                    'breast': 'https://github.com/dfatpnuk/datasets/raw/refs/heads/main/wdbc_discretized.csv'
                    }
        
        results_table = pd.DataFrame(columns=['Test','Dataset','Classifier','N_Instances','N_Features'])  # Initialise the results table
        for dataset in datasets:
            # Load the dataset.
            df = pd.read_csv(dataset_dict[dataset])
            if args.test:  df = df.sample(n=100)   # Runs in testing mode (reduces datasets to 1000 rows.)
            n_rows = len(df)
            n_cols = df.shape[1]

            for classifier in classifiers:

                for max in max_parents:
                    new_row = {
                        'Test': istest, 
                        'Dataset': dataset,
                        'Classifier': classifier,
                        'N_Instances': n_rows,
                        'N_Features': n_cols,
                        'K2_Max_Parents': max
                    }

                    aggregated_results_dict = complete_runs(dataset, df, classifier, max) # Completes 10 runs on df using the current classifier.
                    for key in aggregated_results_dict:
                        new_row[key] = aggregated_results_dict[key]

                    results_table = pd.concat([results_table, pd.DataFrame([new_row])], ignore_index=True)

        # Output the final results table:
        results_table.to_csv(f"results/{datetime.now().strftime('%Y_%m_%d_%H_%M')}_results_table") # Print the results table out to csv.
        results_table_tex = results_table.to_latex(index=False, float_format="%.3f")
        with open(f"results/{datetime.now().strftime('%Y_%m_%d_%H_%M')}_results_table.tex", 'w') as f: f.write(results_table_tex)
        
        print(results_table) # Display final table in the terminal. 
        return
    
    get_results_table(datasets=['breast'],classifiers=['lr'], max_parents=[1,2], istest=args.test)
    get_results_table(datasets=['adult'],classifiers=['lr'], max_parents=[1,2], istest=args.test)
    get_results_table(datasets=['magic'],classifiers=['lr'], max_parents=[1,2], istest=args.test)
    get_results_table(datasets=['breast'],classifiers=['mlp'], max_parents=[1,2], istest=args.test)
    get_results_table(datasets=['adult'],classifiers=['mlp'], max_parents=[1,2], istest=args.test)
    get_results_table(datasets=['magic'],classifiers=['mlp'], max_parents=[1,2], istest=args.test)
    return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some data.")
    parser.add_argument('--test', action='store_true', help='Runs in test mode with reduced data.')
    args = parser.parse_args()

    main(args)
    # bash command to run the tensorboard : tensorboard --logdir runs