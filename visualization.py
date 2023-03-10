import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

if __name__ == '__main__':
    print("Visualizing the dataset")
    real_data = pd.read_csv('Power_grid_real_data.csv').reset_index(drop=True)
    synthetic_data = pd.read_csv("Power_grid_synthetic_data.csv").sample(len(real_data)).sample(len(real_data))
    # Extra column generated by pandas
    del synthetic_data['Unnamed: 0']
    synthetic_data = synthetic_data.drop(['marker'], axis=1)
    real_data = real_data.drop(['marker'], axis=1)
    # ##################################
    # # For base line
    # synthetic_data = real_data.sample(int(len(real_data) * 0.5))
    # real_data = real_data[~real_data.index.isin(synthetic_data.index)]
    ##################################
    real_data['cls'] = "real"
    synthetic_data['cls'] = "synthetic"
    # ##################################################
    whole_data = pd.concat([real_data, synthetic_data])
    scaler = StandardScaler()
    whole_data_scaled = scaler.fit_transform(whole_data.drop(['cls'], axis=1))
    whole_data_embedded = PCA(n_components=2).fit_transform(whole_data_scaled)

    ## creating graphs for four GAN models #########################################
    fig = px.scatter(x=whole_data_embedded[:, 0], y=whole_data_embedded[:, 1], color=whole_data['cls'],
                     color_discrete_sequence=['darkkhaki', 'darkolivegreen'])
    fig.update_traces(textposition='top center')

    fig.update_layout(
        title_text='CTGAN Fine-tuned Model (After Post-process)',
        xaxis_title='Component 1',
        yaxis_title='Component 2',
        paper_bgcolor='rgba(255,255,255,255)',
        plot_bgcolor='rgba(255,255,255,255)'
    )

    fig.update_xaxes(gridcolor='grey', zerolinecolor='grey')
    fig.update_yaxes(gridcolor='grey', zerolinecolor='grey')
    fig.update_layout(legend_title_text='Dataset')
    ##########################################
    # fig = px.scatter(x=whole_data_embedded[:, 0], y=whole_data_embedded[:, 1], color=whole_data['cls'],
    #                  color_discrete_sequence=['burlywood', 'darkgoldenrod'])
    # fig.update_traces(textposition='top center')
    #
    # fig.update_layout(
    #     title_text='CopulaGAN Default Model (Before Post-process)',
    #     xaxis_title='Component 1',
    #     yaxis_title='Component 2',
    #         paper_bgcolor='rgba(255,255,255,255)',
    #         plot_bgcolor='rgba(255,255,255,255)'
    # )
    #
    # fig.update_xaxes(gridcolor='grey',zerolinecolor='grey')
    # fig.update_yaxes(gridcolor='grey',zerolinecolor='grey')
    # fig.update_layout(legend_title_text='Dataset')
    # # ##########################################
    # fig = px.scatter(x=whole_data_embedded[:, 0], y=whole_data_embedded[:, 1], color=whole_data['cls'],
    #                  color_discrete_sequence=['tan', 'sienna'])
    # fig.update_traces(textposition='top center')
    #
    # fig.update_layout(
    #     title_text='TVAE Fine-tuned Model (After Post-process)',
    #     xaxis_title='Component 1',
    #     yaxis_title='Component 2',
    #         paper_bgcolor='rgba(255,255,255,255)',
    #         plot_bgcolor='rgba(255,255,255,255)'
    # )
    #
    # fig.update_xaxes(gridcolor='grey',zerolinecolor='grey')
    # fig.update_yaxes(gridcolor='grey',zerolinecolor='grey')
    # fig.update_layout(legend_title_text='Dataset')
    # # # ##########################################
    # fig = px.scatter(x=whole_data_embedded[:, 0], y=whole_data_embedded[:, 1], color=whole_data['cls'],
    #                  color_discrete_sequence=['mediumseagreen', 'olive'])
    # fig.update_traces(textposition='top center')
    #
    # fig.update_layout(
    #     title_text='GaussianCopula Model (After Post-process)',
    #     xaxis_title='Component 1',
    #     yaxis_title='Component 2',
    #         paper_bgcolor='rgba(255,255,255,255)',
    #         plot_bgcolor='rgba(255,255,255,255)'
    # )
    #
    # fig.update_xaxes(gridcolor='grey',zerolinecolor='grey')
    # fig.update_yaxes(gridcolor='grey',zerolinecolor='grey')
    # fig.update_layout(legend_title_text='Dataset')
    # fig.write_image("CopulaGAN_default_before.png")
    fig.show()
