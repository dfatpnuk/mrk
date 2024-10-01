import sklearn
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.sampling import BayesianModelSampling

# For evaluate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import sys
sys.setrecursionlimit(10000)  # Increase recursion depth to allow for more structures to be evaluated.

import warnings
warnings.filterwarnings("ignore")

class MRK():

    def __init__(self, X = None, y = None):
        """
        self.X - the independent variables.
        self.y - the dependent variable.
        self.data - the entire dataset to be fit to.
        self.variable_names - list of names of all independent variables (X)
        self.target_col_name - the name of the dependent variable (y)
        self.model - the fitted model.
        """
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_data = None
        self.test_data = None
        self.variable_names = None
        self.target_col_name = None
        self.model = None
        self.ordinal_encoder = OrdinalEncoder(dtype=int, handle_unknown='use_encoded_value', unknown_value=-1)
        self.label_encoder = LabelEncoder()

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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Combine X y into train and test sets.
        self.train_data = pd.concat([self.X_train, self.y_train], axis=1)
        self.test_data = pd.concat([self.X_test, self.y_test], axis=1)

        nb_structure = self.naive_bayes()   # Step 1. Initialise Naive Bayes Structure.
        ordered_variables = self.ordered_vars()     # Step 2. Determine an initial ordering for the X variables.

        if verbose: print(f"Ordered variable names: {ordered_variables}")

        self.model = self.k2_structure_learning(ordered_variables, nb_structure, max_parents=2)

        if verbose:
            print("Model has been fit.")
            print(f"Structure Edges: {self.model.edges}")

        return self # Returns self: object (fitted model)

    def naive_bayes(self, data=None, target=None): # Initialise a BN with Naive Bayes. # Accepts the dataset and the target column, returns a Bayesian Network with Naive Bayes structure.
        """
        data - A dataset as a pandas dataframe
        target - The target column name (Y)
        returns: A BayesianNetwork() object with a Naive Bayes structure.
        """
        if data is None: data = self.train_data   # no need to pass in the data itself, it can just be the column names.
        if target is None: target = self.target_col_name

        dependent_vars = data.columns.drop(target)
        edges = [(var, target) for var in dependent_vars]
        model = BayesianNetwork(edges)

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
        #     "arithmetic_mean": lambda mi_score, var_1_entropy, var_2_entropy: 2 * mi_score/ (var_1_entropy + var_2_entropy)}q
        return var_score_dict
        
    def k2_structure_learning(self, ordered_var_names, naive_bayes_bn, max_parents=2):    # Should accept the data and return the optimal structure as a Bayesian Network.
        """
        ordered_var_names - an ordered list of variable names. 
        max_parents - the maximum number of parents allowed for any independent variable. 
        naive_bayes_bn - a BayesianNetwork() object initialised with a Naive Bayes structure.
        """
        visited_vars = []
        best_candidate = copy.deepcopy(naive_bayes_bn) # Init the candidate as the Naive Bayes structure.
        
        for child in ordered_var_names:   # Iterate through the variables in order.
            parents = []
            current_score = self.meg_scoring_function([best_candidate], self.train_data)[0]  # Get the current score
            ok_to_proceed = True
            while len(parents) < max_parents and ok_to_proceed:  # Check if the max number of parents for the variable has been reached.
                
                candidate_structures = []   # Create empty arrays for candidate structures and candidate parents
                candidate_parents = []
                for parent in visited_vars: # Iterate through all the visited variables as candidate parents for the variable. 
                    if parent not in parents:
                        new_candidate = copy.deepcopy(best_candidate)  
                        new_candidate.add_edge(parent, child)
                        candidate_structures.append(new_candidate)  # Add the new structure to the candidate array.
                        candidate_parents.append(parent)

                if not candidate_structures: break  # No candidates to consider

                candidate_scores = self.meg_scoring_function(candidate_structures, self.train_data)
                top_score_idx = candidate_scores.index(max(candidate_scores))   # Find the index of the max candidate score

                if candidate_scores[top_score_idx] > current_score:
                    current_score = candidate_scores[top_score_idx]  # Update the current score
                    best_parent = candidate_parents[top_score_idx]
                    parents.append(best_parent)  # Add the new parent to the parents list
                    best_candidate = candidate_structures[top_score_idx]  # Update the best candidate structure
                else: ok_to_proceed = False  # No improvement, exit the loop

            visited_vars.append(child)  # Add the just visited child to the list of visited variables.

        return best_candidate
        
    def meg_scoring_function(self, array_of_structures, training_data, epochs=30, sample_size=100):  # Scores Bayesian Network structures against the real dataset.
        """
        array_of_structures - An array of Bayesian Network objects to be scored.
        data - the training dataset to which each structure will be fit.
        epochs - Number of epochs for evaluation (default 30).
        sample_size - how many instances to sample from each structure for evaluation
        Returns: An array of length n containing the respective score of each structure.
        """
        array_of_scores = []  # Initialise empty array to house scores. 

        for structure in array_of_structures:  # For each BN in the array fit (fit with the pgmpy) to the training dataset (try 50/50 split to make it less complex)

            structure.fit(training_data, estimator=BayesianEstimator, prior_type='K2')  # Fit the training data to the structure 
            sampler = BayesianModelSampling(structure)
            synthetic_data = sampler.forward_sample(sample_size)  # Generate synthetic data (can be pgmpy forward sampling method, or it can be GANBLR)
            
            # Split synthetic data into X and y
            synthetic_X = synthetic_data[self.variable_names] 
            synthetic_y = synthetic_data[self.target_col_name].astype(int)
            
            # Evaluate synthetic data with TSTR logreg for 30 epochs
            score = self.evaluate(synthetic_X, synthetic_y, self.X_test, self.y_test, epochs=epochs)
            array_of_scores.append(score)   # Populate the array of scores.

        return array_of_scores  # Returns the array of respective scores
    
    def evaluate(self, X_synthetic, y_synthetic, X_real, y_real, epochs) -> float:
        """
        X_synthetic - The X data to be evaluated (X)
        y_synthetic - The y data to be evaluated (y_pred)
        X_real - Dependent variables from the real dataset (X_test)
        y_real - Target variable from the real dataset (y_test)
        """
        # Transform real dataset TODO: Ask Chris why do I need to do this. Commenting out as data has already been encoded...
        # X_real = self.ordinal_encoder.transform(X_real)
        # y_real = self.label_encoder.transform(y_real).astype(int)

        total_accuracy = 0  # Cumulative accuracy score

        for epoch in range(epochs):
            # Set up and train the model pipeline
            pipeline = Pipeline([
                ('encoder', OneHotEncoder(categories='auto', handle_unknown='ignore')),  # Replaced _d.get_categories()  with 'auto'
                ('model', LogisticRegression())])

            pipeline.fit(X_synthetic, y_synthetic)
            pred = pipeline.predict(X_real) # Predict and return accuracy

            accuracy = accuracy_score(y_real, pred)
            total_accuracy += accuracy

        return total_accuracy / epochs

def main(args): # Main function which iterates through the datasets to test the model. 
    """
    Main function which iterates through the datasets to test the model. 
    Prints the results table to the terminal and to a .csv
    """
    if args.test:   # Runs in testing mode (reduces datasets to 1000 rows.)

        # Test using adult dataset from UCIML
        # adult = pd.read_csv('https://raw.githubusercontent.com/chriszhangpodo/discretizedata/main/adult-dm.csv')
        # df = adult.head(100)
        
        # X_train = df.drop('class', axis=1)
        # y_train = df['class']

        # Invent a toy dataset.
        df = pd.DataFrame({
            'test_X1': [0,1,2,3,0,1,2,3,4,3],
            'test_X2': [0,2,2,3,1,2,3,2,1,4],
            'test_X3': [1,2,2,3,1,2,3,2,4,2],
            'test_X4': [0,9,8,7,8,7,6,8,9,7],
            'test_y': [1,2,2,3,1,2,3,2,4,2]
        })

        # Test the fit function
        X_train = df.drop('test_y', axis=1)
        y_train = df['test_y']

        mrk = MRK()
        mrk.fit(X_train, y_train, verbose=True)

    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some data.")
    parser.add_argument('--test', action='store_true', help='Runs in test mode with reduced data.')
    args = parser.parse_args()

    main(args)