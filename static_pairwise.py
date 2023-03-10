import pandas as pd
import statistics
from sdmetrics.column_pairs import CorrelationSimilarity

# 29 per R
if __name__ == '__main__':
    print("evaluating the pairwise similarity ")
    real_data = pd.read_csv('Power_grid_real_data.csv').reset_index(drop=True)
    synthetic_data = pd.read_csv("Power_grid_synthetic_data.csv").sample(len(real_data))
    # Extra column generated by pandas
    del synthetic_data['Unnamed: 0']
    ##################################################

    # The mean value of each of the IEDs is calculated.
    # Note here we are detecting and ignoring columns with a constant value (e.g a column with all 0s)
    r1 = []
    r2 = []
    r3 = []
    r4 = []
    for index1 in range(29):
        for index2 in range(29):
            if ((index1 != index2)):
                if ((len(synthetic_data.loc[synthetic_data[synthetic_data.columns[index1]] == 0]) != len(
                        synthetic_data))
                        and (len(synthetic_data.loc[synthetic_data[synthetic_data.columns[index2]] == 0]) != len(
                            synthetic_data))):
                    r1.append(CorrelationSimilarity.compute(
                        real_data=real_data[[synthetic_data.columns[index1], synthetic_data.columns[index2]]],
                        synthetic_data=synthetic_data[[synthetic_data.columns[index1], synthetic_data.columns[index2]]],
                        coefficient='Spearman'
                    ))
                if ((len(synthetic_data.loc[synthetic_data[synthetic_data.columns[index1 + 29]] == 0]) != len(
                        synthetic_data))
                        and (len(synthetic_data.loc[synthetic_data[synthetic_data.columns[index2 + 29]] == 0]) != len(
                            synthetic_data))):
                    r2.append(CorrelationSimilarity.compute(
                        real_data=real_data[[synthetic_data.columns[index1 + 29], synthetic_data.columns[index2 + 29]]],
                        synthetic_data=synthetic_data[
                            [synthetic_data.columns[index1 + 29], synthetic_data.columns[index2 + 29]]],
                        coefficient='Spearman'
                    ))
                if ((len(synthetic_data.loc[synthetic_data[synthetic_data.columns[index1 + 58]] == 0]) != len(
                        synthetic_data))
                        and (len(synthetic_data.loc[synthetic_data[synthetic_data.columns[index2 + 58]] == 0]) != len(
                            synthetic_data))):
                    r3.append(CorrelationSimilarity.compute(
                        real_data=real_data[[synthetic_data.columns[index1 + 58], synthetic_data.columns[index2 + 58]]],
                        synthetic_data=synthetic_data[
                            [synthetic_data.columns[index1 + 58], synthetic_data.columns[index2 + 58]]],
                        coefficient='Spearman'
                    ))
                if ((len(synthetic_data.loc[synthetic_data[synthetic_data.columns[index1 + 87]] == 0]) != len(
                        synthetic_data))
                        and (len(synthetic_data.loc[synthetic_data[synthetic_data.columns[index2 + 87]] == 0]) != len(
                            synthetic_data))):
                    r4.append(CorrelationSimilarity.compute(
                        real_data=real_data[[synthetic_data.columns[index1 + 87], synthetic_data.columns[index2 + 87]]],
                        synthetic_data=synthetic_data[
                            [synthetic_data.columns[index1 + 87], synthetic_data.columns[index2 + 87]]],
                        coefficient='Spearman'
                    ))
    print("r1", statistics.mean(r1))
    print("r2", statistics.mean(r2))
    print("r3", statistics.mean(r3))
    print("r4", statistics.mean(r4))
