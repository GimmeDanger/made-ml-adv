#!/usr/bin/env python
# coding: utf-8

# # Part-4: EM algorithm
# 
# Теперь главное: ЧГК — это всё-таки командная игра. Поэтому:
# * предложите способ учитывать то, что на вопрос отвечают сразу несколько игроков; скорее всего, понадобятся скрытые переменные; не стесняйтесь делать упрощающие предположения, но теперь переменные “игрок X ответил на вопрос Y” при условии данных должны стать зависимыми для игроков одной и той же команды;
# * разработайте EM-схему для обучения этой модели, реализуйте её в коде;
# * обучите несколько итераций, убедитесь, что целевые метрики со временем растут (скорее всего, ненамного, но расти должны), выберите лучшую модель, используя целевые метрики.
# 

# In[1]:


import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import LinearSVR
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# source: https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file/32216025
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# # Решение
# 
# ### M-step
# * В рамках baseline (part 2) мы научились предсказывать $p(z_{i,j}=1)$ -- вероятность ответил ли игрок i на вопрос j, веса фичей пользователя обученной модели мы интерпретировали как их "силу";
# * Если выбрать $z_{i, j}$ в качестве скрытых переменных, то M-шаг сводится к дообучению модели при заданных $z_(i, j)$, начальные веса совпадают с исходными метками $x_{i,j}$

# In[2]:


# data: describe and computed in part 1
n_epoch = 5
df_train = pd.read_csv('train.zip')
df_train["question_id"] = df_train['tournament_id'].astype(str) + '_' + df_train['question_local_id'].astype(str)
df_train = df_train.drop(columns=['tournament_id', 'question_local_id'])
X, y = df_train[['player_id', 'question_id']], df_train['target']
df_train.head()


# In[7]:


# m-step model: described in part 2

feature_generation = ColumnTransformer(
    transformers=[('OneHot', OneHotEncoder(), ['player_id', 'question_id'])],
    remainder='drop',
    sparse_threshold=1
)

# Замечание: пришлось заменить sklearn::LogisticRegression, тк в ходе выполнения задания
# выяснилось, что она плохо работает с небинарными таргетами, поэтому заменил ее на другой регрессор:
# https://stackoverflow.com/questions/47663569/how-to-do-regression-as-opposed-to-classification-using-logistic-regression-and
pipe = Pipeline(
    verbose=True,
    steps=[
        ('feature_generation', feature_generation),
        ('regressor', LinearSVR(loss='squared_epsilon_insensitive'))
    ]
)

def m_step(model, X, y):
    model.fit(X, y)
    return model, model.predict(X)


# In[8]:


def save_player_weights(X, model):
    """
    сохраняем веса фичей пользователей из обученного классификатора
    """
    player_features_end_pos = X.nunique()['question_id']
    player_features_names = model['feature_generation'].get_feature_names()[0:player_features_end_pos]
    player_ids = [int(name[11:]) for name in player_features_names]
    player_weights = model['regressor'].coef_[0:player_features_end_pos]
    player_to_weight = dict(zip(player_ids, player_weights))
    return player_to_weight

#_, preds = m_step(pipe, X, y)
#validate(load_obj('test'), save_player_weights(X, _))


# # E-step
# 
# * Теперь хотим учесть наличие команды, тч $z_{i,j}$ стали зависимыми для игроков одной команды, перейдем к прогнозированию условных вероятностей $p(z_{i,j} = 1)$ -> $p(z_{i,j} = 1 | team_{i, j} = 1)$, где $team_{i, j}$ -- ответила ли команда игрока i на вопрос j;
# * Предположим, что $$team_{i, j} = 1 \iff \exists k \in team_i : z_{k, j} = 1$$
# * И наоборот: $$team_{i, j} = 0 \iff \forall k \in team_i : z_{k, j} = 0$$

# По теореме Байеса имеем:
# $$
#     p(z_{i,j}=1|team_{i,j}=1) = \frac{p(team_{i,j}=1|z_{i,j}=1) p(z_{i,j}=1)}{p(team_{i,j}=1)}
# $$
# 
# C учетом предположений имеем:
# $$
#     p(z_{i,j}=1|team_{i,j}=1) = \frac{p(z_{i,j}=1)}{1 - p(team_{i,j}=0)} = \frac{p(z_{i,j}=1)}{1 - \Pi_{k \in team_i} \left(1 - p(z_{k,j}=1)\right)}
# $$
# 
# С учетом того, что $p(z_{i,j}=1)$ являются результатом M-шага, то формула выше может быть использована для E-шага

# In[9]:


def e_step(df, preds):
    df['new_target'] = preds
    label_zero_idx = df['target'] == 0
    df.loc[label_zero_idx, 'new_target'] = 0
    # изменяем только метки для вопросов, на которые команда ответила
    # поскольку p(z_ij = 1 | team_ij = 0) = 0 в силу предположений
    label_one_idx = df['target'] == 1
    e_step_denom = df.loc[label_one_idx].groupby(['team_id', 'question_id'])['new_target']
    e_step_denom = e_step_denom.transform(lambda x : 1 - np.prod(1 - x.values))
    df.loc[label_one_idx, 'new_target'] = df.loc[label_one_idx, 'new_target'] / e_step_denom
    new_y = df['new_target'].fillna(0)
    return new_y


# In[10]:


# initialization
pipe, preds = m_step(pipe, X, y)
#validate(load_obj('test'), save_player_weights(X, pipe))


# In[ ]:


# EM-iterations
for i in range(n_epoch):
    y = e_step(df_train, preds)
    pipe, preds = m_step(pipe, X, y)
    weights = save_player_weights(X, pipe)
    save_obj(weights, f'em_weights_epoch_{i}')


# In[12]:


# validation: described in part 3

def get_positions_label(tournament):
    """
    позиции команд в турнире (фактические)
    """
    return [team['position'] for team in tournament]


def get_position_prediction(tournament, player_to_weight):
    """
    позиции команд в турнире (предсказанные),
    ранжируем команды по весу = (сумма весов участников),
    есть игрока не было в train -- берем средний вес игрока в трейне
    """
    avg_weight = np.mean([v for v in player_to_weight.values()])
    team_rating = []
    for idx, team in enumerate(tournament):
        weight = 0
        for player_info in team['teamMembers']:
            p_id = player_info['player']['id']
            try:
                weight += player_to_weight[p_id]
            except:
                weight += avg_weight
        team_rating.append((idx + 1, weight))
    team_rating = sorted(team_rating, key=lambda kv: kv[1], reverse=True)
    return [pos for pos, weight in team_rating]


def get_score(df_test, player_to_weight, corr):
    """
    среднее значение rank correlation по тестовой выборке
    """
    x = [corr(get_positions_label(t), get_position_prediction(t, player_to_weight)).correlation for t in df_test.values()]
    x = np.array(x)
    x = x[~np.isnan(x)]
    return np.mean(x)

def validate(df_test, player_to_weight, corr=None):
    if corr is None:
        for corr in [('Spearman', stats.spearmanr), ('Kendall ', stats.kendalltau)]:
            print(f'Avg {corr[0]} corr value for df = {get_score(df_test, player_to_weight, corr[1])}')
    else:
        print(f'Avg {corr[0]} corr value for df = {get_score(df_test, player_to_weight, corr[1])}')


# In[ ]:


for i in range(n_epoch):
    print(f'Epoch {i+1}:')
    validate(load_obj('test'), load_obj(f'em_weights_epoch_{i}'))


# # Метрики не растут. В чем дело? Ход разбирательства:
# 
# * Бейзлайн модель LogisticRegression, метрики падали сильнейшим образом, выяснил что данная модель из sklearn некорректно работает с небинарными таргетами, а именно они возникают на итерациях EM-алгоритм;
# * Заменил модель на LinearSVR, настроил параметры: метрики все также не растут, но хотя бы перестали убывать;

# In[ ]:




