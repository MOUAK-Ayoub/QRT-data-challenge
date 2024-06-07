import pandas as pd


def prepare_attribut_data(path_away: str, path_home: str):
    df_away = pd.read_csv(path_away).drop(['LEAGUE', 'TEAM_NAME'], axis=1)
    df_home = pd.read_csv(path_home).drop(['LEAGUE', 'TEAM_NAME'], axis=1)
    df_home.columns = [('HOME_' + str(col)) if col != 'ID' else 'ID' for col in df_home.columns]
    df_joined = df_away.join(df_home.set_index('ID'), on='ID')
    df_joined = df_joined.fillna(0.0)
    df_final = df_joined.sort_values(by=['ID'])

    return df_final


def prepare_result_data(path_result):
    df_result = pd.read_csv(path_result)
    df_result.loc[df_result['DRAW'] == 1, 'result'] = 0
    df_result.loc[df_result['AWAY_WINS'] == 1, 'result'] = 1
    df_result.loc[df_result['HOME_WINS'] == 1, 'result'] = 2
    df_result['result'] = df_result['result'].astype('int')
    df_result_final = df_result.sort_values(by=['ID']).iloc[:, [4]]

    return df_result_final

def convert_to_one_hot(yhat_predicted):
    y_pred_test = pd.DataFrame(yhat_predicted)
    y_pred_test.columns = ['prediction']
    y_pred_test.loc[y_pred_test['prediction'] == 1, 'HOME_WINS'] = 1
    y_pred_test.loc[y_pred_test['prediction'] == 0, 'DRAW'] = 1

    y_pred_test.loc[y_pred_test['prediction'] == -1, 'AWAY_WINS'] = 1

    y_pred_test = y_pred_test.fillna(0)
    y_pred_test['HOME_WINS'] = y_pred_test['HOME_WINS'].astype('int')
    y_pred_test['DRAW'] = y_pred_test['DRAW'].astype('int')
    y_pred_test['AWAY_WINS'] = y_pred_test['AWAY_WINS'].astype('int')
    y_pred_test = y_pred_test.drop(['prediction'], axis=1)
    return y_pred_test

