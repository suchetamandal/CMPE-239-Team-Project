
# coding: utf-8

# In[ ]:

import sqlite3
import pandas as pd
from time import time
import warnings
warnings.simplefilter("ignore")


# In[ ]:

database = 'database.sqlite'
conn = sqlite3.connect(database)


# In[ ]:

player_data = pd.read_sql("SELECT * FROM Player;", conn)
player_stats_data = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
team_data = pd.read_sql("SELECT * FROM Team;", conn)
match_data = pd.read_sql("SELECT * FROM Match;", conn)
league_data = pd.read_sql("SELECT * from League;",conn)
country_data = pd.read_sql("SELECT * from Country;",conn)
team_more_data = pd.read_sql("SELECT * from Team_Attributes;",conn)


# In[ ]:

## Loading all functions
match_data.drop('BWA', axis=1, inplace=True)
match_data.drop('IWH', axis=1, inplace=True)
match_data.drop('IWD', axis=1, inplace=True)
match_data.drop('IWA', axis=1, inplace=True)
match_data.drop('LBH', axis=1, inplace=True)
match_data.drop('LBD', axis=1, inplace=True)
match_data.drop('LBA', axis=1, inplace=True)
match_data.drop('PSH', axis=1, inplace=True)
match_data.drop('PSD', axis=1, inplace=True)
match_data.drop('PSA', axis=1, inplace=True)
match_data.drop('WHH', axis=1, inplace=True)
match_data.drop('WHD', axis=1, inplace=True)
match_data.drop('WHA', axis=1, inplace=True)
match_data.drop('SJH', axis=1, inplace=True)
match_data.drop('VCA', axis=1, inplace=True)
match_data.drop('VCH', axis=1, inplace=True)
match_data.drop('VCD', axis=1, inplace=True)
match_data.drop('GBH', axis=1, inplace=True)
match_data.drop('GBD', axis=1, inplace=True)
match_data.drop('GBA', axis=1, inplace=True)
match_data.drop('BSH', axis=1, inplace=True)
match_data.drop('BSD', axis=1, inplace=True)
match_data.drop('BSA', axis=1, inplace=True)
match_data.drop('date', axis=1, inplace=True)
match_data.drop('B365D', axis=1, inplace=True)
match_data.drop('B365A', axis=1, inplace=True)
match_data.drop('BWD', axis=1, inplace=True)
match_data.drop('BWH', axis=1, inplace=True)
match_data.drop('possession', axis=1, inplace=True)
match_data.drop('corner', axis=1, inplace=True)
match_data.drop('goal', axis=1, inplace=True)
match_data.drop('shoton', axis=1, inplace=True)
match_data.drop('shotoff', axis=1, inplace=True)
match_data.drop('foulcommit', axis=1, inplace=True)
match_data.drop('card', axis=1, inplace=True)
match_data.drop('cross', axis=1, inplace=True)
match_data.drop('B365H', axis=1, inplace=True)
match_data.drop('season', axis=1, inplace=True)


# In[ ]:

## Lebel Creation ('Result') by comparing 'home_team_goal' and 'away_team_goal', is Win (+1) if home_team_goal > away_team_goal
## Lost('-1') if away_team_goal > home_team_goal and Draw ('0') if away_team_goal = home_team_goal
def get_result_home_advantage(match_data):
    results = []
    for match in range(len(match_data.index)):
        home_goals = match_data['home_team_goal'].iloc[match]
        away_goals = match_data['away_team_goal'].iloc[match] 
        
        if home_goals > away_goals:
            results.append('+1')
        elif away_goals > home_goals:
            results.append('-1')
        else:
            results.append('0')
    match_data['result'] = results 
    return match_data        


# In[ ]:

start = time()
match_rev = get_result_home_advantage(match_data) 
end = time()
print("Result column added in {:.1f} seconds".format((end - start)))


# In[ ]:

##Away Match and Home Match Winner Bucket Creation
away_match_winner =  match_rev.loc[match_rev['result'].isin(['-1'])]
home_match_winner =  match_rev.loc[match_rev['result'].isin(['+1'])]
draw_match = match_rev.loc[match_rev['result'].isin(['0'])]


# In[ ]:

## create one featured for telling whether a player is attacking or not.
#Feature Reduction and also binoritization
#x = player_stats_data['attacking_work_rate']
#y = player_stats_data['defensive_work_rate']

#Medium means 0 None means None, High means 1 (attacker) , (defensive)
def get_player_type(player_data):
    attacks = []
    i = 0
    defenses = []
    j = 0
    midfield = []
    k = 0
    for player in range(len(player_data.index)):
        attacking = player_data['attacking_work_rate'].iloc[player]
        defensing = player_data['defensive_work_rate'].iloc[player]
        
        if attacking == 'high':
            attacks.append('1')
            i = i+ 1
        else:
            attacks.append('0')
            
        if defensing == 'high':
            defenses.append('1')
            j = j+1
        else:
            defenses.append('0')
        
        if attacking == 'medium' or defensing == 'medium':
            midfield.append('1')
            k = k+1
        else:
            midfield.append('0')
            
            
    player_data['attacker'] = attacks
    player_data['defender'] = defenses
    player_data['midfielder'] = midfield
    
    print("Attackers are "+str(i))
    print("Defenders are "+str(j))
    print("MidFielders are "+str(k))
    
    return player_data 

def get_free_kick_player(player_data):
    freekicks = []
    i = 0;
    j = 0;
    for player in range(len(player_data.index)):
        kick_accuracy = player_data['free_kick_accuracy'].iloc[player]
        vision = player_data['vision'].iloc[player]
        kick_vision = ((kick_accuracy*0.80)+(vision*0.20))/200
        if kick_vision > 0.35:
            freekicks.append('1')
            i = i+1
        else:
            freekicks.append('0')
            j = j + 1
    player_data['free_kick_taker'] = freekicks 
    print("Free Kick Taker is "+str(i))
    print("Non Free Kick Taker is "+str(j))
    return player_data     


# In[ ]:

py1= get_free_kick_player(player_stats_data)
player_stats_updated = get_player_type(py1)


# In[226]:

player_stats_updated['gk_diving'].fillna(0)
player_stats_updated['gk_handling'].fillna(0)
player_stats_updated['gk_kicking'].fillna(0)
player_stats_updated['gk_positioning'].fillna(0)
player_stats_updated['gk_reflexes'].fillna(0)

player_stats_updated.to_csv('goal.csv',encoding='utf-8')


# In[236]:

def getPlayerAllStat(player_id):
    if np.isnan(player_id) == True:
        print('No Data')
        return 0,0,0,0,0
    else:
        #home_home_games = result_df[(result_df.home_team_api_id == home_team)].tail(games)
        player = player_stats_updated[(player_stats_updated.player_api_id == player_id)].tail(1)
        #for index,player in player_stats_updated.iterrows():
            #if(player['player_api_id'] == player_id): 
        if player['gk_diving'].isnull().any: 
            x=0
        else:    
            x = player['gk_diving']
            
        if player['gk_handling'].isnull().any: 
            y=0
        else:    
            y = player['gk_handling']
            
        if player['gk_kicking'].isnull().any: 
            z=0
        else:    
            z = player['gk_kicking']
         
        if player['gk_positioning'].isnull().any: 
            a=0
        else:    
            a = player['gk_positioning']
            
        if player['gk_reflexes'].isnull().any: 
            b=0
        else:    
            b = player['gk_reflexes']
            
        goali = x+y+z+a+b
        
        if player['free_kick_taker'].isnull().any: 
            freekick=0
        else:    
            freekick = player['free_kick_taker']
            
        if player['attacker'].isnull().any: 
            attack=0
        else:    
            attack = player['attacker']
            
        if player['defender'].isnull().any: 
            defend=0
        else:    
            defend = player['defender']
            
        if player['midfielder'].isnull().any: 
            midfielder=0
        else:    
            midfielder = player['midfielder']    
        return goali,freekick,attack,defend,midfielder


# In[242]:

def getPlayerStatsForAllTeams(match_data):
    goalkeeper_home_score = []
    goalkeeper_away_score = []
    
    attacking_home_score = []
    attacking_away_score = []
    
    defending_home_score = []
    defending_away_score = []
    
    midfield_home_score = []
    midfield_away_score = []
    
    freekick_home_score = []
    freekick_away_score =[]
    i=0
    for match in range(len(match_data.index)):
        away_player = set()
        home_player = set()
        
        hp1 = match_data['home_player_1'].iloc[match]
        ap1 = match_data['away_player_1'].iloc[match]

        
        hp2 = match_data['home_player_2'].iloc[match]
        ap2 = match_data['away_player_2'].iloc[match]
        
        hp3 = match_data['home_player_3'].iloc[match]
        ap3 = match_data['away_player_3'].iloc[match]
        
        hp4 = match_data['home_player_4'].iloc[match]
        ap4 = match_data['away_player_4'].iloc[match]
    
        
        home_player.add(hp1)
        home_player.add(hp2)
        home_player.add(hp3)
        home_player.add(hp4)

        
        away_player.add(ap1)
        away_player.add(ap2)
        away_player.add(ap3)
        away_player.add(ap4)
        
        g1=0
        g2=0
        f1=0
        f2=0
        a1=0
        a2=0
        d1=0
        d2=0
        m1=0
        m2=0
        for hp in home_player:
            goali,freekick,attack,defend,midfielder = getPlayerAllStat(hp)
            g1 = max(g1,int(goali))
            f1 = f1 + int(freekick)
            a1 = a1 + int(attack)
            d1 = d1 + int(defend)
            m1 = m1 + int(midfielder)
        for ap in away_player:
            goali,freekick,attack,defend,midfielder = getPlayerAllStat(hp)
            g2 = max(g2,int(goali))
            f2 = f1 + int(freekick)
            a2 = a1 + int(attack)
            d2 = d1 + int(defend)
            m2 = m1 + int(midfielder)

        goalkeeper_home_score.append(g1)
        goalkeeper_away_score.append(g2)
        freekick_home_score.append(f1)
        freekick_home_score.append(f2)
        attacking_home_score.append(a1)
        attacking_away_score.append(a2)
        defending_home_score.append(d1)
        defending_away_score.append(d2)
        midfield_home_score.append(m1)
        midfield_away_score.append(m2)
        
        i=i+1
        print('Match Evaluated'+str(i))
        
    match_data['home_goali'] = goalkeeper_home_score
    match_data['away_goali'] = goalkeeper_away_score
    match_data['home_attacker'] = attacking_home_score
    match_data['away_attacker'] = attacking_away_score
    match_data['home_defender'] = defending_home_score
    match_data['away_defender'] = defending_away_score
    match_data['home_midfielder'] = midfield_home_score
    match_data['away_midfielder'] = midfield_away_score
    return match_data


# In[243]:

start = time()
length = (len(match_data.index))
match_data_updated = getPlayerStatsForAllTeams(match_rev)
end = time()
print("Result column added in {:.1f} seconds".format((end - start)))


# In[244]:

def get_last_matches_against_eachother(matches, date, home_team, away_team, x = 10):
    ''' Get the last x matches of two given teams. '''
    
    #Find matches of both teams
    home_matches = matches[(matches['home_team_api_id'] == home_team) & (matches['away_team_api_id'] == away_team)]    
    away_matches = matches[(matches['home_team_api_id'] == away_team) & (matches['away_team_api_id'] == home_team)]  
    total_matches = pd.concat([home_matches, away_matches])
    
    #Get last x matches
    try:    
        last_matches = total_matches[total_matches.date < date].sort_values(by = 'date', ascending = False).iloc[0:x,:]
    except:
        last_matches = total_matches[total_matches.date < date].sort_values(by = 'date', ascending = False).iloc[0:total_matches.shape[0],:]
        
        #Check for error in data
        if(last_matches.shape[0] > x):
            print("Error in obtaining matches")
            
    #Return data
    return last_matches


# In[249]:

#Simple KNN on match data
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.cross_validation import train_test_split
import pandas as pd 

match_data_updated.dropna(inplace = True)
labels = match_data_updated.loc[:,'result']
features = match_data_updated.drop('result', axis = 1)

clf1 = neighbors.KNeighborsClassifier(n_neighbors=1000)
clf2 = neighbors.KNeighborsClassifier(n_neighbors=2000)
clf3 = neighbors.KNeighborsClassifier(n_neighbors=3000)
clf4 = neighbors.KNeighborsClassifier(n_neighbors=4000)
clf5 = neighbors.KNeighborsClassifier(n_neighbors=5000)
clf6 = neighbors.KNeighborsClassifier(n_neighbors=6000)
clf7 = neighbors.KNeighborsClassifier(n_neighbors=7000)
clf8 = neighbors.KNeighborsClassifier(n_neighbors=8000)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2)

clf1.fit(X_train,y_train)
accuracy1 = clf1.score(X_test,y_test)

clf2.fit(X_train,y_train)
accuracy2 = clf2.score(X_test,y_test)

clf3.fit(X_train,y_train)
accuracy3 = clf3.score(X_test,y_test)

clf4.fit(X_train,y_train)
accuracy4 = clf4.score(X_test,y_test)

clf5.fit(X_train,y_train)
accuracy5 = clf5.score(X_test,y_test)

clf6.fit(X_train,y_train)
accuracy6 = clf6.score(X_test,y_test)

clf7.fit(X_train,y_train)
accuracy7 = clf7.score(X_test,y_test)

clf8.fit(X_train,y_train)
accuracy8 = clf8.score(X_test,y_test)
print(accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6,accuracy7,accuracy8)

from sklearn.naive_bayes import GaussianNB

clfnb = GaussianNB()
clfnb.fit(X_train, y_train)
accuracyNB = clfnb.score(X_test,y_test)

print("In Gaussian NB")
print (accuracyNB)


##WITH TruncatedSVD + KNN
from sklearn.decomposition import PCA, FastICA,TruncatedSVD
from sklearn.pipeline import Pipeline
trun = TruncatedSVD()
dm_reductions = [trun]  
clf_details = [clf]
estimators = [('dm_reduce', trun), ('clf', clf)]
pipeline = Pipeline(estimators)        
best_pipe = pipeline.fit(X_train, y_train)
bestAccuracy = pipeline.score(X_test,y_test)
print("In KNN plus Trunkcated")
print(bestAccuracy)

##WITH PCA + KNN
from sklearn.decomposition import PCA, FastICA,TruncatedSVD
from sklearn.pipeline import Pipeline
pca = PCA()
dm_reductions = [pca]  
clf_details = [clf]
estimators = [('dm_reduce', pca), ('clf', clf)]
pipeline = Pipeline(estimators)        
best_pipe = pipeline.fit(X_train, y_train)
bestAccuracy = pipeline.score(X_test,y_test)
print("In KNN plus PCA")
print(bestAccuracy)


# In[ ]:

0.395897435897 20% data
0.395897435897 30% data
0.460697197539 10% data k = 1000
0.455042735043 20% data k = 1000
0.464957264957 20% data k = 3000
0.468717948718 20% data k = 5000
0.46188034188 20% data k = 10000
0.478290598291 20% data k = 6000


# In[ ]:



