#IMPORTS

from Bio import SeqIO
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn import model_selection, linear_model

import gdown
data_path = 'https://drive.google.com/uc?id=1f1CtRwSohB7uaAypn8iA4oqdXlD_xXL1'
cov2_sequences = 'SARS_CoV_2_sequences_global.fasta'
gdown.download(data_path, cov2_sequences, True)

print("\n\n*******IMPORTS COMPLETE*******\n\n")





#DATA PREPROCESSING

sequences = [r for r in SeqIO.parse(cov2_sequences, 'fasta')]
#choose index to display
sequence_num =  0 
print(sequences[sequence_num])

#print number of sequences
n_sequences = len(sequences) 
print("There are %f sequences" % n_sequences)

#how different are the 1st (non-reference) and 10th SARS-CoV-2 sequences?
sequence_1 = np.array(sequences[1])
sequence_10 = np.array(sequences[100])
percent_similarity = sum(sequence_1 == sequence_10) / len(sequence_1)
print("Sequence 1 and 10 similarity: %", percent_similarity)





#FEATURE EXTRACTION: X

# Note: This can take a couple minutes to run! 
# but we can monitor our progress using the tqdm library
mutation_df = pd.DataFrame()
n_bases_in_seq = len(sequences[0])

# Iterate though all positions in this sequence.
for location in tqdm.tqdm(range(n_bases_in_seq)): # tqdm is a nice library that prints our progress.
  bases_at_location = np.array([s[location] for s in sequences])
  # If there are no mutations at this position, move on.
  if len(set(bases_at_location))==1: continue # If
  for base in ['A', 'T', 'G', 'C', '-']:
    feature_values = (bases_at_location==base)
    
    # Set the values of any base that equals 'N' to np.nan.
    feature_values[bases_at_location=='N'] = np.nan
    
    # Convert from T/F to 0/1.
    feature_values  = feature_values*1
    
    # Make the column name look like <location>_<base> (1_A, 2_G, 3_A, etc.)
    column_name = str(location) + "_" + base
    mutation_df[column_name] = feature_values

# Print the size of the feature matrix/table.
n_rows = np.shape(mutation_df)[0]
n_columns = np.shape(mutation_df)[1]
print("Size of matrix: %i rows x %i columns" %(n_rows, n_columns))

# Check what the matrix looks like:
mutation_df.head()



#FEATURE EXTRACTION: Y

#convert each country to its region of the world
countries_to_regions_dict = {
         'Australia':   'Oceania',
         'China':       'Asia',
         'Hong Kong':   'Asia',
         'India':       'Asia' ,
         'Nepal':       'Asia',
         'South Korea': 'Asia' ,
         'Sri Lanka':   'Asia' ,
         'Taiwan':      'Asia' ,
         'Thailand':    'Asia' ,
         'USA':         'North America' ,
         'Viet Nam':    'Asia' 
}

countries = [(s.description).split('|')[-1] for s in sequences]

regions = [countries_to_regions_dict[c] if c in 
           countries_to_regions_dict else 'NA' for c in countries]
mutation_df['label'] = regions



#BALANCING THE DATA

balanced_df = mutation_df.copy()
balanced_df['label'] = regions
balanced_df = balanced_df[balanced_df.label!='NA']
balanced_df = balanced_df.drop_duplicates()
samples_north_america = balanced_df[balanced_df.label== 'North America']
samples_oceania = balanced_df[balanced_df.label== 'Oceania']
samples_asia = balanced_df[balanced_df.label== 'Asia']

# Number of samples we will use from each region.
n = min(len(samples_north_america), len(samples_oceania), len(samples_asia))

df_balanced = pd.concat([samples_north_america[:n],
                    samples_asia[:n],
                    samples_oceania[:n]])
print("Number of samples in each region: ", Counter(df_balanced['label']))



#LOGISTIC REGRESSION MODEL

#make matricies
X = balanced_df.drop('label', 1)
Y = balanced_df.label
data = "X (features)"
start = 1
stop =  20

if start>=stop:print("Start must be < stop!")
else:
  if data=='X (features)':
    print(X.iloc[start:stop])
  if data=='Y (label)':
    print(Y[start:stop])


#TRAIN
lm = linear_model.LogisticRegression(
multi_class="multinomial", max_iter=1000,
fit_intercept=False, tol=0.001, solver='saga', random_state=42)

# Split into training/testing set. Use a training size of .8
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 101)

# Train/fit model.
lm.fit(X_train, Y_train)


#TESTING
# Predict on the test set.
Y_pred = lm.predict(X_test)

# Compute accuracy.
cm = confusion_matrix(Y_test, Y_pred)
TP = cm[0][0]
TN = cm[1][1]
FP = cm[0][1]
FN = cm[1][0]
accuracy = (TP + TN)/(TP + TN + FP + FN)
print("Accuracy: %", accuracy)

# Compute confusion matrix.
confusion_mat = pd.DataFrame(confusion_matrix(Y_test, Y_pred))
confusion_mat.columns = [c + ' predicted' for c in lm.classes_]
confusion_mat.index = [c + ' true' for c in lm.classes_]

print(confusion_mat)