import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
This one is trying to analyze the status change (account / ebill / activation) people

and its floor ratio (floor / total call). Also, percentages of status change variations are also recorded

Output of the code is written in comments

"""
df = pd.read_csv('data_n.csv')

df_change_account_stat = df[df['account_status_13_march'] != df['account_status_18_march']]
resolved_counts_account = df_change_account_stat['resolved'].value_counts()
print(f"resolved_counts_account = {resolved_counts_account}")
df_change_ebill_stat = df[df['ebill_enrolled_status_13_march'] != df['ebill_enrolled_status_18_march']]
resolved_counts_ebill = df_change_ebill_stat['resolved'].value_counts()
print(f"resolved_counts_ebil = {resolved_counts_ebill}")
df_change_activation = df[df['card_activation_status_13_march'] != df['card_activation_status_18_march']]
resolved_counts_activation = df_change_activation['resolved'].value_counts()
print(f"resolved_counts_activation = {resolved_counts_activation}")

"""
Result for the above chunk:
 floor ratio on account_status change = 70.8%
 floor ratio ebill stat change = 40.7%
 floor ratio activation change = 36.2%
"""

# this is to see usually account status changed from what to what
# filtering to floored people only
df_change_account_stat = df_change_account_stat[df_change_account_stat['resolved'] == 'floor']
# Create a new column to encode the value change
df_change_account_stat['change'] = df_change_account_stat['account_status_13_march'] + ' to ' + df_change_account_stat['account_status_18_march']
# Count the number of occurrences of each change
change_counts_account = df_change_account_stat['change'].value_counts()
# Calculate the total number of occurrences
total_occurrences_account = change_counts_account.sum()
# Calculate the percentages
change_percentages_account = (change_counts_account/ total_occurrences_account) * 100
# Displaying the counts
print(change_percentages_account)

"""
result (total people who changed in account status, including Floor/Resolve)
PERCENTAGE
change
N to L    45.744058
N to C    33.456270
N to A    12.542442
A to N     5.947781
A to L     1.404988
A to C     0.295633
L to U     0.210748
C to L     0.108301
C to N     0.105374
N to U     0.067322
N to B     0.061468
B to C     0.020489
C to U     0.014635
C to A     0.008781
C to B     0.005854
U to C     0.002927
B to N     0.002927

result (FLOOR'd change_account_status)
change
N to L    50.028947
N to C    26.771979
N to A    14.233728
A to N     6.157473
A to L     1.840212
A to C     0.359772
L to U     0.165412
C to L     0.128195
C to N     0.090977
N to U     0.086841
N to B     0.070300
B to C     0.028947
C to U     0.016541
C to A     0.008271
C to B     0.008271
U to C     0.004135
"""

# filtering to floored people only
df_change_ebill_stat = df_change_ebill_stat[df_change_ebill_stat['resolved'] == 'floor']
# Create a new column to encode the value change
df_change_ebill_stat['change'] = df_change_ebill_stat['ebill_enrolled_status_13_march'] + ' to ' + df_change_ebill_stat['ebill_enrolled_status_18_march']
# Count the number of occurrences of each change
change_counts_ebill = df_change_ebill_stat['change'].value_counts()
# Calculate the total number of occurrences
total_occurrences_ebill = change_counts_ebill.sum()
# Calculate the percentages
change_percentages_ebill = (change_counts_ebill/ total_occurrences_ebill) * 100
# Displaying the counts
print(change_percentages_ebill)

"""
for ebill change (total people who changed in ebill status, including Floor/Resolve)
change
N to B    39.667038
B to E    21.925918
N to E    19.178221
E to N    12.640421
E to B     3.724319
B to N     2.864083

for ebill change (Floord people only)
N to B    31.497344
N to E    24.579855
B to E    22.563157
E to N    13.412122
E to B     4.911634
B to N     3.035889
"""
# filtering to floored people only
df_change_activation = df_change_activation[df_change_activation['resolved'] == 'floor']
# Create a new column to encode the value change
df_change_activation['change'] = df_change_activation['card_activation_status_13_march'].astype(str) + ' to ' + df_change_activation['card_activation_status_18_march'].astype(str)
# Count the number of occurrences of each change
change_counts_activation = df_change_activation['change'].value_counts()
# Calculate the total number of occurrences
total_occurrences_activation = change_counts_activation.sum()
# Calculate the percentages
change_percentages_activation = (change_counts_activation/ total_occurrences_activation) * 100
# Displaying the counts
print(change_percentages_activation)
"""
change (total people who changed in activation status, including Floor/Resolve)
0 to 8    46.698565
7 to 0    23.053757
0 to 7    10.095694
8 to 0     9.594709
0 to 1     4.002252
1 to 8     2.780749
1 to 0     2.352941
7 to 8     1.049817
8 to 1     0.354630
1 to 7     0.014073
8 to 7     0.002815

change in activation status ,Floor'd people only
0 to 8    38.443580
7 to 0    27.439689
0 to 7    12.972763
8 to 0    11.914397
0 to 1     3.081712
1 to 8     2.926070
1 to 0     1.968872
7 to 8     1.003891
8 to 1     0.225681
1 to 7     0.015564
8 to 7     0.007782
"""
