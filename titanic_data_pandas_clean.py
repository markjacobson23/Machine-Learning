import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def clean(filepath,ans=True):
  df = pd.read_csv(filepath)
  print(df.isna().sum())
  df['Title']=df['Name'].str.extract(r",\s*([^\.]+)\.")
  title_to_sex = {'Mr':'Male', 'Mrs':'Female', 'Miss':'Female', 'Master':'Male', 'Don':'Male', 'Rev':'Male', 'Dr':'Male', 'Mme':'Female', 'Ms':'Female', 'Major':'Male', 'Lady':'Female',
   'Sir':'Male', 'Mlle':'Female', 'Col':'Male', 'Capt':'Male', 'the Countess':'Female', 'Jonkheer':'Male','Dona':'Female'}
  df['Sex']=df['Title'].map(title_to_sex)
  df['Sex Binary'] = df['Sex'].map({'Male':0,'Female':1})

  df['Age']=df['Age'].fillna(df['Age'].mode()[0])
  df['Deck'] = df['Cabin'].str.extract(r"\s*([\w])\w*")
  missing_firstclass_cabins= df.loc[df['Pclass'] == 1,'Deck'].isna().sum()
  df.loc[(df['Pclass'] == 1) & (df['Deck'].isna()) ,"Deck"] = np.random.choice(["A", "B", "C","D","E"], size=missing_firstclass_cabins, p=[ 36/365,101/365,134/365,49/365,45/365])
  missing_secondclass_cabins= df.loc[df['Pclass'] == 2,'Deck'].isna().sum()
  df.loc[(df['Pclass'] == 2) & (df['Deck'].isna()) ,"Deck"] = np.random.choice(["D","E",'F'], size=missing_secondclass_cabins, p=[39/168,65/168,64/168])
  missing_thirdclass_cabins= df.loc[df['Pclass'] == 3,'Deck'].isna().sum()
  df.loc[(df['Pclass'] == 3) & (df['Deck'].isna()) ,"Deck"] = np.random.choice(["D","E","F","G"], size=missing_thirdclass_cabins, p=[11/222,74/222,115/222,22/222])
  df['UpperDeck'] = ((df['Deck'] == 'A') | (df['Deck'] == 'B') | (df['Deck'] == 'C')).astype(int)
  df['MiddleDeck'] = ((df['Deck'] == 'D') | (df['Deck'] == 'E')).astype(int)
  df['Embarked'] = df['Embarked'].fillna('S')
  df['EmbarkS'] = (df['Embarked'] == 'S').astype(int)
  df['EmbarkC'] = (df['Embarked'] == 'C').astype(int)
  print(df.isna().sum())
  print(df[df['Sex'].isna()].index)
  df = df.fillna(0)
  cleaned_data = df[['Pclass','Age','SibSp','Parch','Fare','EmbarkS','EmbarkC','Sex Binary','UpperDeck','MiddleDeck']].to_numpy()
  targets = df[['Survived']].to_numpy() if ans == True else None
  pids = df[['PassengerId']].to_numpy()
  return cleaned_data, targets, pids

clean_data, target,pidx = clean('train.csv')
target = target.flatten()
pidx = pidx.flatten()
means = clean_data[:,:5].mean(axis=0)
dev = clean_data[:,:5].std(axis=0)
clean_data[:,:5] = (clean_data[:,:5]-means)/dev
test_data, dumm, tpidx = clean('test.csv',ans = False)
means = test_data[:,:5].mean(axis=0)
dev = test_data[:,:5].std(axis=0)
test_data[:,:5] = (test_data[:,:5]-means)/dev
tpidx = tpidx.flatten()
model = LogisticRegression(max_iter=10000000, solver='liblinear',tol=.0000000000001)
model.fit(clean_data,target)
print(model.predict(test_data))

submission = pd.DataFrame({
    "PassengerId": tpidx,
    "Survived": model.predict(test_data)
})

submission.to_csv("submission.csv", index=False)





