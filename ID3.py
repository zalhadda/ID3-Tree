###############
### IMPORTS ###
###############

import numpy as np
import pandas as pd
import statistics
import sys
import os

##################
### TREE CLASS ###
##################
    
class Node(object):
    
    '''
    Inputs:
        - feature: attribute of current node
        - split_value: median value of feature for split
        - survived_count: # survived
        - dead_count: # not survived
        - right: right leaf
        - left: left leaf
        - depth: depth of node
    '''
    
    def __init__(self, 
                 examples, 
                 feature, 
                 survived_count, 
                 dead_count, 
                 split_value, 
                 label = None, 
                 right = None, 
                 left = None, 
                 depth = None):

        self.examples = examples
        
        self.feature = feature        
        self.split_value = split_value
        self.label = label
        
        self.survived_count = survived_count
        self.dead_count = dead_count
        
        self.right = right
        self.left = left
        
        self.depth = depth
        
        self.parent = None

# End of class Node

class Tree(object):
    
    def __init__(self):
        self.root = None
        
    # Returns the root node
        
    def Get_Root(self):
        return self.root
    
    '''
    Inputs:
        - feature: attribute of current node
        - split_value: median value of feature for split
        - survived_count: # survived
        - dead_count: # not survived
        - right: right leaf
        - left: left leaf
        - depth: depth of node
    '''
    
    def Add(self, node):
        
        # If there's no root, set node to root
        
        if (self.root == None):
            node.depth = 1 # root is the first node in the tree
            self.root = node
                    
        else:
            self._Add(node, prev_node = self.root)
            
            
    def _Add(self, node, prev_node):
    
        # Value less than the parent node.split_value -> goes to left leaf
        
        if (node.examples[prev_node.feature].unique()[0] < prev_node.split_value):
            
            # If there prev node doesn't have a left leaf,
            #   make this the left leaf
            # Else: Keep going down the tree
            
            if (prev_node.left != None):
                self._Add(node, prev_node.left)
                
            else:
                # Update the node's depth 
                
                node.depth = prev_node.depth + 1
                
                # Set the parent_node's left to node
                
                prev_node.left = node
                
                # Set the node's parent to prev_node
                
                node.parent = prev_node
                
                                
        # Value greater than or equal to parent node.split_value -> goes to right leaf
        
        else:
            
            # If there prev node doesn't have a right leaf,
            #   make this the right leaf
            # Else: Keep going down the tree
            
            if (prev_node.right != None):
                self._Add(node, prev_node.right)
                
            else:
                # Update the node's depth
                
                node.depth = prev_node.depth + 1
                
                # Set the parent_node's right pointer to node
                
                prev_node.right = node
                
                # Set the node's parent to prev_node
                
                node.parent = prev_node
                
                        
# End of class Tree
                
########################
### GLOBAL VARIABLES ###
########################

# Instantiates a tree class

tree = Tree()


##########################
### ID3 TREE FUNCTIONS ###
##########################

'''
Creates the ID3 tree.
* inpsired by the pseudo code at https://en.wikipedia.org/wiki/ID3_algorithm

Input:
    - tree: instantiated tree
    - examples: dataset to build tree
    - features: attributes of the dataset
    - target_feature: feature that acts as the label
    - max_depth: maximum depth of tree, if None == no max depth specified
    - parent_node: the previous node in the tree
'''	

def ID3(tree, 
        examples, 
        features, 
        target_feature, 
        parent_node = None,
        max_depth = None,
        min_split = None):
    
    # Counts
    
    total_count = examples.count()[0]
    survived_count = examples[examples[target_feature] == 1].count()[0]
    dead_count = examples[examples[target_feature] == 0].count()[0]
    
    # If the max_depth field is specified, check for the max_depth of the tree
    # Once the max_depth is received, assign a label to the target_feature
    #   with the highest count

    if (max_depth != None and parent_node != None):
                    
        # If the parent_node's depth is one less than the max_depth,
        # then this next node will be the maximum depth of the
        #   current section of the tree
        
        # Label will be decided based on which target_feature value is highest
        
        if (parent_node.depth == (max_depth - 1)):
            
            if (survived_count >= dead_count):
                
                # Create the node
                
                node = Node(examples = examples,
                         feature = target_feature,
                         split_value = None,
                         label = 1,
                         survived_count = survived_count,
                         dead_count = dead_count,
                         right = None,
                         left = None)
                
                # Add the node to the tree
                
                tree.Add(node)
            
            else:
                
                # Create the node
                
                node = Node(examples = examples,
                         feature = target_feature,
                         split_value = None,
                         label = 0,
                         survived_count = survived_count,
                         dead_count = dead_count,
                         right = None,
                         left = None)
                
                # Add the node to the tree
                
                tree.Add(node)
    
            return    
    
    # Base Case: If all examples have survived = 1, return a node with '1'
    
    if (survived_count == total_count):
        
        # Create the node
        
        node = Node(examples = examples,
                 feature = target_feature,
                 label = 1,
                 survived_count = survived_count,
                 dead_count = dead_count,
                 split_value = None)
                    
        # Add the node to the tree
        
        tree.Add(node)
        
        return
        
    # Base Case: If all examples have survived = 0, return a node with '0'
    
    if (dead_count == total_count):
        
        # Create the node
        
        node = Node(examples = examples,
                 feature = target_feature,
                 label = 0,
                 survived_count = survived_count,
                 dead_count = dead_count,
                 split_value = None)
        
        # Add the node to the tree
        
        tree.Add(node)
        
        return
        
    # Base Case: If there are no more features,
    #   add a leaf node with label = greatest target_feature value
    
    if (len(features) == 0):
        if (survived_count >= dead_count):
            
            # Create the node
            
            node = Node(examples = examples,
                     feature = target_feature,
                     label = 1,
                     survived_count = survived_count,
                     dead_count = dead_count,
                     split_value = None)
            
            # Add the node to the tree
            
            tree.Add(node)
        
        else:
            
            # Create the node
            
            node = Node(examples = examples,
                     feature = target_feature,
                     label = 0,
                     survived_count = survived_count,
                     dead_count = dead_count,
                     split_value = None)
            
            # Add the node to the tree
            
            tree.Add(node)
    
        return
        
    # Calculates the information gains for every feature
    
    info_gains = Calculate_Features_Info_Gain(examples, features)    
    
    # Find the feature with the largest information gain
    
    feature_with_max_gain = None
    prev_value = -1
    
    for key, value in info_gains.items():
        
        # If the min_split flag is set, 
        #   then a feature needs to have at least min_split values
        #   to be considered for a split
        
        if (min_split != None):
            if (examples[key].count() < min_split):
                continue
        
        if (value > prev_value):
            prev_value = value
            feature_with_max_gain = key
            
    # If no features match the min_split, then feature_with_max_gain is None
    
    if (feature_with_max_gain == None):
        
        # Add a leaf node with the higher survivor/dead count as the label
        
        if (survived_count >= dead_count):
            
            # Create the node
            
            node = Node(examples = examples,
                     feature = target_feature,
                     label = 1,
                     survived_count = survived_count,
                     dead_count = dead_count,
                     split_value = None)
            
            # Add the node to the tree
            
            tree.Add(node)
        
        else:
            
            # Create the node
            
            node = Node(examples = examples,
                     feature = target_feature,
                     label = 0,
                     survived_count = survived_count,
                     dead_count = dead_count,
                     split_value = None)
            
            # Add the node to the tree
            
            tree.Add(node)
    
        return
    
    # Remove the feature_with_max_gain from the features list
    # This updated list will be used in the recursive step
    
    updated_features = []
    
    for feature in features:
        
        if (feature != feature_with_max_gain):
            updated_features.append(feature)
            
        # Only removing the discrete features
        # Continuous features can remain
        #   Fare is the only float
        #       The rest are ints or boolean
        # Age is set as continous for this assignment even tho it's an int
        #   because it was split multiple times in the homework instructions
            
        if (feature == "Fare"
            or feature == "Age"):
            
            updated_features.append(feature)
    
    # Splits the feature
    
    feature_median = statistics.median(examples[feature_with_max_gain].unique())
    
    # Create the node
    
    node = Node(examples = examples, 
              feature = feature_with_max_gain, 
              split_value = feature_median, 
              survived_count = survived_count, 
              dead_count = dead_count)
    
    # Add the node to the tree
    
    tree.Add(node)
    
    # Retrieve the examples that are above than the median and below the median
    
    examples_greater_than_median = examples[examples[feature_with_max_gain] >= feature_median]
    examples_less_than_median = examples[examples[feature_with_max_gain] < feature_median]
    
    # If the # of examples in either group is 0,
    # add a leaf node with the label as the most common target_feature value
    
    if (len(examples_greater_than_median.index) == 0
        or
        len(examples_less_than_median.index) == 0):
        
        if (survived_count >= dead_count):
            
            # Create the node
            
            node = Node(examples = examples,
                     feature = target_feature,
                     label = 1,
                     survived_count = survived_count,
                     dead_count = dead_count,
                     split_value = None)
            
            # Add the node to the tree
            
            tree.Add(node)
        
        else:
            
            # Create the node
            
            node = Node(examples = examples,
                     feature = target_feature,
                     label = 0,
                     survived_count = survived_count,
                     dead_count = dead_count,
                     split_value = None)
            
            # Add the node to the tree
            
            tree.Add(node)
            
        return
        
    # Recursive Step
    # Creates two leaves for every node,
    #   Each leaf will represent the values on either side of the split value
    
    ID3(tree = tree, 
        examples = examples_greater_than_median, 
        features = updated_features, 
        target_feature = target_feature,
        parent_node = node,
        max_depth = max_depth,
        min_split = min_split)
    
    ID3(tree = tree, 
        examples = examples_less_than_median, 
        features = updated_features, 
        target_feature = target_feature,
        parent_node = node,
        max_depth = max_depth,
        min_split = min_split)
    
# End of ID3()


'''
Tests for the accuracy of the ID3 tree

Inputs:
    - tree: the constructed tree
    - data: the dataset to test the tree on
    - features: attributes of the testing dataset
    
Outputs:
    - The accuracy as a float
'''

def Find_Accuracy_ID3(tree, data, features):
    
    features = features.tolist()
    
    # Initialise the root and crawler variables
    # crawler will traverse the tree
    # root will be used to reset the crawler
    
    tree_root = tree.root
    tree_crawler = tree_root

    # Initialise the counts
    
    total_examples = data.count()[0]
    total_correct_labels = 0
        
    # Iterate through every row in the data
    # Each row will have the following format:
    #   Pclass, Sex, Age, Fare, Embarked, relatives, IsAlone, Survived
    
    for row in data.itertuples(index = False, name = 'Pandas'):
                        
        while ((tree_crawler.label == None)
            or (tree_crawler.left != None and tree_crawler.right != None)):
                                    
            # Find the value of the water feature feature
            
            feature_index = features.index(tree_crawler.feature)
            feature_value = row[feature_index]
            
            # If the value is less than the split_value,
            #   go to the left leaf
            
            if (feature_value < tree_crawler.split_value):
                
                # Break if there are no more nodes
                
                if (tree_crawler.left == None):
                    break
                else:
                    tree_crawler = tree_crawler.left
                
            # If the value is greater than or equal to the split_value
            #   go to the right leaf
            
            else:
                
                # Break of there are no more nodes
                
                if (tree_crawler.right == None):
                    break
                else:
                    tree_crawler = tree_crawler.right
                
        # Check if the label in row is the same as in tree_crawler
        
        expected_label = row[-1]
        actual_label = tree_crawler.label
        
        #print("Expected: " + str(expected_label) + "\nActual: " + str(actual_label) + "\n")
        
        if (str(expected_label) == str(actual_label)):
            total_correct_labels += 1
            
        # Reset the tree_crawler 
        
        tree_crawler = tree_root
                    
    # Calculate the accuracy
    
    accuracy = total_correct_labels / total_examples
    
    return round(accuracy, 4)
    
# End of Find_Accuracy_ID3()

'''
Prunes the given ID3 tree.
Pruning stops once the validation accuracy is lower than the 
    prev_validation_accuracy 
    (i.e. the greatest increase in validation accuracy is found)

Inputs:
    - tree: ID3 tree
    - node: ID3 binary node
    - dataset: validation dataset to be used for pruning
    - features: features of the dataset
    - prev_validation_accuracy: accuracy of the previous pruning iteration
'''

def Prune(tree,
          node, 
          dataset, 
          features,
          prev_validation_accuracy = None):
    
    # If there's no current prev_validation_accuracy,
    #   Find the accuracy for the current tree
    
    if (prev_validation_accuracy == None):
        prev_validation_accuracy = Find_Accuracy_ID3(tree, dataset, features)
        
    # Turn the node into a leaf by setting its label to the higher of
    #   survived_count or dead_count and the branches to None
    
    # Set label
    
    if (node.survived_count >= node.dead_count):
        node.label = 1
    else:
        node.label = 0
        
        
    # Delete branches
    
    temp_left_branch = node.left
    temp_right_branch = node.right
    
    node.left = None
    node.right = None
        
    # Find the accuracy with the current node as leaf
    
    validation_accuracy = Find_Accuracy_ID3(tree, dataset, features)
    
    # If the validation_accuracy is more than the prev_validation_accuracy,
    #   then prune the tree
    #  else, unprune it
    
    if (validation_accuracy >= prev_validation_accuracy):
        return #pruned
    
    # Unprune node
    
    node.label = None
    node.left = temp_left_branch
    node.right = temp_right_branch
    
    # Recursively iterate through the branches of node
    
    if (node.left != None):
        Prune(node = node.left, 
              dataset= dataset, 
              features = features, 
              prev_validation_accuracy = validation_accuracy, 
              tree = tree)
        
    if (node.right != None):
        Prune(node = node.right, 
              dataset= dataset, 
              features = features, 
              prev_validation_accuracy = validation_accuracy, 
              tree = tree)
    
# End of Prune()

########################
### HELPER FUNCTIONS ###
########################

'''
Prints the script's usage case
'''

def Print_Usage_Case():
    
    print("Usage: <script> <training_data_path> <testing_data_path>"
                  + " <decision_tree_type> <decision_tree_arguments>\n")
        
    print("Types of decision trees: \n"
              + "- \"vanilla\" <training_set_percentage>\n"
              + "- \"depth\" <training_set_percentage> "
                  + "<validationg_set_percentage> <maximum_depth>\n"
              + "- \"min_split\" <training_set_percentage> "
                  + "<validationg_set_percentage> <minimum_splits>\n"
              + "- \"prune\" <training_set_percentage> "
                  + "<validationg_set_percentage>\n"
        )
        
# End of Print_Usage_Case()
    
'''
Calculates the Entropy of the given frequencies.
Entropy = -SUM (Pi * log(Pi)) for all i

Input: A list of size 2 where 
        [0] = # positive results, [1] = # negative results
        
Output: A float representing the entropy of the given frequencies
'''

def Entropy(freqs):
    """ 
    Example:
    entropy(p) = -SUM (Pi * log(Pi))
    >>> entropy([10.,10.])
    1.0
    >>> entropy([10.,0.])
    0
    >>> entropy([9.,3.])
    0.811278
    """
    
    all_freq = sum(freqs)
    entropy = 0 
    
    for fq in freqs:
        
        if (all_freq == 0):
            prob = 0
        else:
            prob = fq * 1.0 / all_freq
        
        if abs(prob) > 1e-8:
            entropy += -prob * np.log2(prob)
            
    return entropy

# End of Entropy()

'''
Calculates the information gain.
Information_Gain(D, A) = Entropy(D) - SUM( |Ai| / |D| * Entropy(Ai) )
    where D is a list of [positive_results, negative_results] before splitting
    and A is a list of a list of [positive_results, negative_results] after
        splitting for each feature.
        
Input: Two lists equivalent to (D, A). See above.

Output: A float representing the information again.
'''    

def Information_Gain(before_split_freqs, after_split_freqs):
    """
    Example:
    gain(D, A) = Entropy(D) - SUM ( |Ai| / |D| * Entropy(Ai) )
    >>> infor_gain([9,5], [[2,2],[4,2],[3,1]])
    0.02922
    """
    
    gain = Entropy(before_split_freqs)
    overall_size = sum(before_split_freqs)
    
    for freq in after_split_freqs:
        
        if (overall_size != 0):
            ratio = sum(freq) * 1.0 / overall_size
        else:
            ratio = 0
            
        gain -= ratio * Entropy(freq)
        
    return gain

# End of Information_Gain()
    
'''
Finds the depth of a given tree

Inputs:
    - node: tree's root node
    
Outputs:
    - An init specifiying the tree's depth
'''

def Tree_Depth(node):
    
    # No root node = tree is empty, return 0
    
    if (node == None):
        return 0
    
    # Recursively finds the depth of the right & left branches
    
    left_branch_depth = Tree_Depth(node.left)
    right_branch_depth  = Tree_Depth(node.right)
    
    # Return the larger depth
    # Adding one to account for the 0 that's returned for the leaf nodes
    
    if (left_branch_depth > right_branch_depth):
        return (left_branch_depth + 1)
    else:
        return (right_branch_depth + 1)
    
    
'''
Returns the num_nodes of the tree
'''

def Num_Nodes(node):
        
    # No node = not counted
    
    if (node == None):
        return 0
    
    # Recursively move through the tree and add up the # of nodes
    
    return 1 + Num_Nodes(node.left) + Num_Nodes(node.right)
    
# End of Num_Nodes()

##################
### UNIT TESTS ###
##################
    
def Entropy_Unit_Tests():
    
    ''' Entropy Unit Test 1 '''

    test_1 = [0.5, 0.5]
    expected_1 = 1.0
    result_1 = Entropy(test_1)
    
    if ( str(round(expected_1, 5)) == str(round(result_1, 5)) ):
        print("Entropy Unit Test 1: Passed\n")
        
    else:
        print("Entropy Unit Test 1: Failed")
        print("\tExpected:", expected_1, "\n\tResult:", result_1, "\n")
    
    ''' Entropy Unit Test 2 '''
    
    test_2 = [1.0, 0.0]
    expected_2 = 0.0
    result_2 = Entropy(test_2)
    
    if ( str(round(expected_2, 5)) == str(round(result_2, 5)) ):
        print("Entropy Unit Test 2: Passed\n")
        
    else:
        print("Entropy Unit Test 2: Failed")
        print("\tExpected:", expected_2, "\n\tResult:", result_2, "\n")
    
    ''' Entropy Unit Test 3 '''
    
    test_3 = [0.75, 0.25]
    expected_3 = 0.811278
    result_3 = Entropy(test_3)
    
    if ( str(round(expected_3, 5)) == str(round(result_3, 5)) ):
        print("Entropy Unit Test 3: Passed\n")
        
    else:
        print("Entropy Unit Test 3: Failed")
        print("\tExpected:", expected_3, "\n\tResult:", result_3, "\n")
    
    
# End of Entropy_Unit_Tests
        
def Information_Gain_Unit_Tests():
    
    ''' Information Gain Unit Test 1 '''

    test_1_before = [9, 5]
    test_1_after = [ [2, 2], [4, 2], [3, 1] ]
    expected_1 = 0.02922
    result_1 = Information_Gain(test_1_before, test_1_after)
    
    if ( str(round(expected_1,5)) == str(round(result_1,5)) ):
        print("Information Gain Unit Test 1: Passed\n")
        
    else:
        print("Information Gain Unit Test 1: Failed")
        print("\tExpected:", expected_1, "\n\tResult:", result_1, "\n")
    
    
# End of Information_Gain_Unit_Tests()
        
'''
Reads a given data file and returns the relevant information.
Features returned: <Pclass> <Sex> <Age> <Relatives> <IsAlone> <Fare> <Embarked>

Input: File to data
Output: A pandas dataframe corresponding to the read file
'''

def Read_Data_File(data_file):

    # Reads the data from the csv file
    
    data = pd.read_csv(data_file, delimiter = ',', index_col = None, 
                       engine = 'python')
    
    return data

    
# End of Read_Data_File()

'''
Reads a given label file and returns all the data in the file in a list

Input: File to labels
Output: List of labels, each index corresponding to the relevant line
'''

def Read_Labels_File(labels_file):
    
    # Reads the labels  from the csv file
    
    labels = pd.read_csv(labels_file, delimiter = ',', index_col = None, 
                       engine = 'python')
    
    return labels

# End of Read_Label_File()

'''
Calculates the information gain for every feature in the given dataset

Inputs:
    - dataset: A pandas dataframe containing all the data
    - features: The attributes of the dataset
    
Output: A dictionary with the features' information gains
'''    

def Calculate_Features_Info_Gain(dataset, features):

    # Retrieves the total counts of who survived and who did not
    
    total_freq_survivors = [dataset[dataset['survived'] == 1].count().unique()[0], 
                       dataset[dataset['survived'] == 0].count().unique()[0]]
    
    # Finds the information gain for each feature in dataset
    
    info_gains = dict()
    
    for feature in features:
        
        # Find the median of the values for a binary split
 
        feature_median = statistics.median(dataset[feature].unique())
        
        # Splits the data into two clusters based on the median
        
        feature_freq_survivors = []
        
        # Greater than or equal to the median
        
        feature_greater_than_median = dataset[dataset[feature] >= feature_median]
        feature_freq = [feature_greater_than_median[feature_greater_than_median['survived'] == 1].count().unique()[0],
                        feature_greater_than_median[feature_greater_than_median['survived'] == 0].count().unique()[0]]
        
        feature_freq_survivors.append(feature_freq)
        
        # Less than to the median
        
        feature_less_than_median = dataset[dataset[feature] < feature_median]
        feature_freq = [feature_less_than_median[feature_less_than_median['survived'] == 1].count().unique()[0],
                        feature_less_than_median[feature_less_than_median['survived'] == 0].count().unique()[0]]
        
        feature_freq_survivors.append(feature_freq)
        
        # Calculates the information gain for the feature
        
        feature_info_gain = Information_Gain(total_freq_survivors, feature_freq_survivors)
        info_gains[feature] = feature_info_gain
                    
    return info_gains

############
### MAIN ###
############

def main(args):
    
    global tree
    tree = Tree()    
    
    # Runs unit tests
    # Set to 0 to turn off tests
    # Set to 1 to turn on tests
    
    if (0):
        Entropy_Unit_Tests()
        Information_Gain_Unit_Tests()
        
    # Parse command line arguments
    # Command line arguments are expected to take the form of:
    #   <python script> <training data> <testing data> <type of decision tree> 
    #       <arguments pertaining to the decision tree>
    
    # Handles invalid number of parameters and prints the usage case
    
    if (len(args) < 5) or (len(args) > 7):
        
        print("Invalid number of parameters.\n")
        Print_Usage_Case()
        sys.exit()
        
    # Assigns the decision tree variable and training set percentage
        
    type_of_decision_tree = str(args[3])
    training_set_percentage = int(args[4])


    # Checks that the trainining and testing data paths are valid directories
    # If they are, appends the relevant training & testing file names for
    #   both the data files and the labels files
    
    train_data_path = ""
    train_labels_path = ""
    test_data_path = ""
    test_labels_path = ""
    
    if ( os.path.isdir(str(args[1])) ):
        train_data_path = str(args[1]) + "/titanic-train.data"
        train_labels_path = str(args[1]) + "/titanic-train.label"
    
    else:
        print("Invalid training data directory.\n")
        sys.exit()
    
    if ( os.path.isdir(str(args[2])) ):
        test_data_path = str(args[2]) + "/titanic-test.data"
        test_labels_path = str(args[2]) + "/titanic-test.label"
        
    else:
        print("Invalid testing data directory.\n")
        sys.exit()
        
    # Read the training and testing data
    # train_data and test_data are pandas data frames
    
    train_data = Read_Data_File(train_data_path)
    test_data = Read_Data_File(test_data_path)
        
    # Read corresponding training and testing labels
    # train_labels and test_labels are pandas data frames
    
    train_labels = Read_Labels_File(train_labels_path)
    test_labels = Read_Labels_File(test_labels_path)    
    
    # Merges the train_data and train_labels
    # Merges the test_data and test_labels
    
    train = train_data.join(train_labels)
    test = test_data.join(test_labels)
    
    # Defines the train and test features
    # i.e. Pclass, Sex, Age, Relatives, IsAlone, Fare, Embarked
    
    train_features = train.columns[:-1]
    test_features = test.columns[:-1]
    
    # Reduces the training set to the percentage of dataset specified for 
    #   training in the command-line arguments
    
    # Adds a to_train column
    # training_set_percentage of the data is assigned to be trained with
    # The remaining (100% - training_set_percentage) will be removed
    
    train['to_train'] = np.random.uniform(0, 1, len(train)) <= (training_set_percentage / 100)
    to_train = train[train['to_train'] == True]

    # to_not_train will be used to create validation sets  
    
    to_not_train = train[train['to_train'] == False]
    
    # Removes the to_train column
    
    train = to_train.drop('to_train', 1)
    to_not_train = to_not_train.drop('to_train', 1)    
    
    # Builds the specified decision tree
    
    ### VANILLA TREE ###
    
    if (type_of_decision_tree == "vanilla"):
                
        # Create the ID3 tree
        #   tree = instantiated Tree class
        #   train = training dataset
        #   train_features = features of the training dataset
        #   target_feature = 'survived' in this case
        
        ID3(tree = tree, 
            examples = train, 
            features = train_features, 
            target_feature = 'survived')
                
        # Test the tree's accuracy for training and testing datasets
        
        train_accuracy = Find_Accuracy_ID3(tree, train, train_features)
        test_accuracy = Find_Accuracy_ID3(tree, test, test_features)
        
        # Print out accuracies
        
        print("Train set accuracy: " + str(train_accuracy))
        print("Test set accuracy: " + str(test_accuracy))
        
        #  Returns information for statistical purposes
        
        return [train_accuracy, test_accuracy, Num_Nodes(tree.root)]
                
    ### MAX DEPTH TREE ###
        
    elif (type_of_decision_tree == "depth"):
                
        validation_set_percentage = int(args[5])
        maximum_depth = int(args[6])
        
        # Creates the validation dataset
                
        to_not_train['to_validate'] = np.random.uniform(0, 1, len(to_not_train)) <= (validation_set_percentage / 100)
        validate = to_not_train[to_not_train['to_validate'] == True]
        validate = validate.drop('to_validate', 1)
        
        validate_features = validate.columns[:-1]
                
        # Create the ID3 tree
        #   tree = instantiated Tree class
        #   train = training dataset
        #   train_features = features of the training dataset
        #   target_feature = 'survived' in this case
        #   max_depth = maximum depth of the tree
        
        ID3(tree = tree, 
            examples = train, 
            features = train_features, 
            target_feature = 'survived',
            max_depth = maximum_depth)
        
        # Test the tree's accuracy for training, validation, and testing datasets
        
        train_accuracy = Find_Accuracy_ID3(tree, train, train_features)
        validate_accuracy = Find_Accuracy_ID3(tree, validate, validate_features)
        test_accuracy = Find_Accuracy_ID3(tree, test, test_features)
        
        # Print out accuracies
        
        print("Train set accuracy: " + str(train_accuracy))
        print("Validation set accuracy: " + str(validate_accuracy))
        print("Test set accuracy: " + str(test_accuracy))
        
        #  Returns information for statistical purposes
        
        return [train_accuracy, validate_accuracy, test_accuracy, Num_Nodes(tree.root)]
        
    ### MIN SPLIT TREE ###
        
    elif (type_of_decision_tree == "min_split"):
                
        validation_set_percentage = int(args[5])
        minimum_split = int(args[6])
        
        # Creates the validation dataset
                
        to_not_train['to_validate'] = np.random.uniform(0, 1, len(to_not_train)) <= (validation_set_percentage / 100)
        validate = to_not_train[to_not_train['to_validate'] == True]
        validate = validate.drop('to_validate', 1)
        
        validate_features = validate.columns[:-1]
        
        # Create the ID3 tree
        #   tree = instantiated Tree class
        #   train = training dataset
        #   train_features = features of the training dataset
        #   target_feature = 'survived' in this case
        #   min_split = minimum # of samples a feature needs to be split
        
        ID3(tree = tree, 
            examples = train, 
            features = train_features, 
            target_feature = 'survived',
            min_split = minimum_split)
        
        # Print out accuracies
        
        print("Train set accuracy: " + str(train_accuracy))
        print("Validation set accuracy: " + str(validate_accuracy))
        print("Test set accuracy: " + str(test_accuracy))
            
    ### PRUNE TREE ###
    
    elif (type_of_decision_tree == "prune"):
                        
        validation_set_percentage = int(args[5])
        
        # Creates the validation dataset
                
        to_not_train['to_validate'] = np.random.uniform(0, 1, len(to_not_train)) <= (validation_set_percentage / 100)
        validate = to_not_train[to_not_train['to_validate'] == True]
        validate = validate.drop('to_validate', 1)
        
        validate_features = validate.columns[:-1]
        
        # Create the ID3 tree
        #   tree = instantiated Tree class
        #   train = training dataset
        #   train_features = features of the training dataset
        #   target_feature = 'survived' in this case
        
        ID3(tree = tree, 
            examples = train, 
            features = train_features, 
            target_feature = 'survived')
                
        # Prune left subtree

        left_most_node = tree.root
        
        while(left_most_node.left != None):
            left_most_node = left_most_node.left
            
        while(left_most_node.parent != tree.root):
                        
            Prune(node = left_most_node.parent,
                  dataset = validate,
                  features = validate_features,
                  tree = tree)
            
            left_most_node = left_most_node.parent

        # Prune right subtree            
            
        right_most_node = tree.root
        
        while(right_most_node.left != None):
            right_most_node = right_most_node.left
            
        while(right_most_node.parent != tree.root):
            
            Prune(node = right_most_node.parent,
                  dataset = validate,
                  features = validate_features,
                  tree = tree)
            
            right_most_node = right_most_node.parent
        
        # Test the tree's accuracy for training and testing datasets
        
        train_accuracy = Find_Accuracy_ID3(tree, train, train_features)
        test_accuracy = Find_Accuracy_ID3(tree, test, test_features)
        
        # Print out accuracies
        
        print("Train set accuracy: " + str(train_accuracy))
        print("Test set accuracy: " + str(test_accuracy)) 

        #  Returns information for statistical purposes
        
        return [train_accuracy, test_accuracy, Num_Nodes(tree.root)]
        
    else:
        
        print("Invalid type of decision tree detected.\n")
        Print_Usage_Case()
        sys.exit()
        
    
# End of main()

##################
##################
    
if __name__=='__main__':
    
    arguments = []
    
    for arg in sys.argv:
        arguments.append(arg)
        
    sys.exit(main(arguments))
        