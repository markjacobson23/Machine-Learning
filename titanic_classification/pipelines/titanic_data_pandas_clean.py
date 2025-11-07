import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def clean(filepath,train=True):
  df = pd.read_csv(filepath)
  print(df.isna().sum())

  # Extract titles from names with regex
  df['Title']=df['Name'].str.extract(r",\s*([^\.]+)\.")

  # Generate sex column from title mappings
  title_to_sex = {'Mr':'Male', 'Mrs':'Female', 'Miss':'Female', 'Master':'Male', 'Don':'Male', 'Rev':'Male', 'Dr':'Male', 'Mme':'Female', 'Ms':'Female', 'Major':'Male', 'Lady':'Female',
   'Sir':'Male', 'Mlle':'Female', 'Col':'Male', 'Capt':'Male', 'the Countess':'Female', 'Jonkheer':'Male','Dona':'Female'}
  df['Sex']=df['Title'].map(title_to_sex)

  # One-Hot encode sex to binary
  df['Sex Binary'] = df['Sex'].map({'Male':0,'Female':1})

  # Fill age NAs with mode (technically first element of mode)
  df['Age']=df['Age'].fillna(df['Age'].mode()[0])

  # Extract Deck letter from cabin string
  df['Deck'] = df['Cabin'].str.extract(r"\s*([\w])\w*")

  # Fill missing decks with a rough distribution of decks according to class.
  # This uses the rough layout of the titanic deck-class associations
  missing_firstclass_cabins= df.loc[df['Pclass'] == 1,'Deck'].isna().sum()
  df.loc[(df['Pclass'] == 1) & (df['Deck'].isna()) ,"Deck"] = np.random.choice(["A", "B", "C","D","E"], size=missing_firstclass_cabins, p=[ 36/365,101/365,134/365,49/365,45/365])
  missing_secondclass_cabins= df.loc[df['Pclass'] == 2,'Deck'].isna().sum()
  df.loc[(df['Pclass'] == 2) & (df['Deck'].isna()) ,"Deck"] = np.random.choice(["D","E",'F'], size=missing_secondclass_cabins, p=[39/168,65/168,64/168])
  missing_thirdclass_cabins= df.loc[df['Pclass'] == 3,'Deck'].isna().sum()
  df.loc[(df['Pclass'] == 3) & (df['Deck'].isna()) ,"Deck"] = np.random.choice(["D","E","F","G"], size=missing_thirdclass_cabins, p=[11/222,74/222,115/222,22/222])

  # Encode decks into upper, middle, (lower is 0,0)
  df['UpperDeck'] = ((df['Deck'] == 'A') | (df['Deck'] == 'B') | (df['Deck'] == 'C')).astype(int)
  df['MiddleDeck'] = ((df['Deck'] == 'D') | (df['Deck'] == 'E')).astype(int)

  # Fill embarked NAs with most common port of departure
  df['Embarked'] = df['Embarked'].fillna('S')

  # Encode ports of departure to binary
  df['EmbarkS'] = (df['Embarked'] == 'S').astype(int)
  df['EmbarkC'] = (df['Embarked'] == 'C').astype(int)

  print(df.isna().sum())
  print(df[df['Sex'].isna()].index)

  # Fill negligible remaining NAs with 0 (all fields are numerical at this point)
  df = df.fillna(0)

  # Normalize non-binary fields
  cleaned_data = df[['Pclass','Age','SibSp','Parch','Fare','EmbarkS','EmbarkC','Sex Binary','UpperDeck','MiddleDeck']].to_numpy()
  means = cleaned_data[:,:5].mean(axis=0)
  dev = cleaned_data[:,:5].std(axis=0)
  cleaned_data[:,:5] = (cleaned_data[:,:5]-means)/dev

  # Return cleaned data and targets if ans=True
  targets = df[['Survived']].to_numpy().flatten() if train == True else None
  pids = df[['PassengerId']].to_numpy().flatten()
  return cleaned_data, targets, pids

# Model block
clean_data, target, pidx = clean('../data/train.csv',train=True)
test_data, dumm, tpidx = clean('test.csv',train = False)
model = LogisticRegression(max_iter=100000, solver='liblinear',tol=.0001)
model.fit(clean_data,target)








