import pgmpy as pgm
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df  = pd.read_csv("data/BN/BN_discrete.csv",   sep=",", header=0)
df_train, df_test = train_test_split(df, test_size=0.1) #500 rows for validating

df_train_target = df_train["Winner"]
df_train_features = df_train.drop("Winner", axis=1)
df_test_target = df_test["Winner"]
df_test_features = df_test.drop("Winner", axis=1)

#for autocomplete
atk_def = "Atk/Def ratio"
sp_atk_def = "Sp.Atk/Def ratio"
def_atk = "Def/Atk ratio"
sp_def_atk = "Sp.Def/Atk ratio"
HP = "HP advantage"
Type = "Type Advantage"
Speed = "Speed advantage"
OA = "Offensive advantage"
OR = "Offensive ratio"
DA = "Defensive advantage"
DR = "Defensive ratio"
Winner = "Winner"

G = BayesianModel()
G.add_nodes_from(df.columns)
G.add_nodes_from([OA, DA, OR, DR])
edges = [(atk_def, OR), (sp_atk_def, OR), (Speed, OA), (OR, OA), (def_atk, DR), (sp_def_atk, DR), (HP, DA), (DR, DA), (Type, Winner), (OA, Winner), (DA, Winner)]
for edge in edges:
    G.add_edge(edge[0], edge[1])

cpd_atk_def = TabularCPD(atk_def, 4, [[0.25], [0.25], [0.25], [0.25]])
cpd_sp_atk_def = TabularCPD(sp_atk_def, 4, [[0.25], [0.25], [0.25], [0.25]])
cpd_def_atk = TabularCPD(def_atk, 4, [[0.25], [0.25], [0.25], [0.25]])
cpd_sp_def_atk = TabularCPD(sp_def_atk, 4, [[0.25], [0.25], [0.25], [0.25]])
cpd_HP = TabularCPD(HP, 4, [[0.25], [0.25], [0.25], [0.25]])
cpd_speed = TabularCPD(Speed, 4, [[0.25], [0.25], [0.25], [0.25]])
cpd_type = TabularCPD(Type, 3, [[0.3], [0.4], [0.3]])

### DOMAIN KNOWLEDGE WOOOOHOOOOOooooo

                                #--                  -                    +                  ++
                                #--    -    +   ++   --    -    +   ++   --   -    +    ++   --    -    +   ++
cpd_OA = TabularCPD(OR, 4,     [[0.5, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],   # very bad
                                [0.3, 0.2, 0.4, 0.4, 0.2, 0.5, 0.4, 0.3, 0.4, 0.4, 0.1, 0.2, 0.4, 0.2, 0.2, 0.1],   # bad
                                [0.1, 0.2, 0.2, 0.4, 0.2, 0.1, 0.4, 0.4, 0.3, 0.4, 0.5, 0.2, 0.4, 0.4, 0.2, 0.3],   # good
                                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.3, 0.5, 0.1, 0.3, 0.5, 0.5]],  # very good
                                evidence=[atk_def, sp_atk_def], evidence_card=[4, 4])

cpd_OR = TabularCPD(OA, 4,     [[0.5, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],   # very bad
                                [0.3, 0.2, 0.4, 0.4, 0.2, 0.5, 0.4, 0.3, 0.4, 0.4, 0.1, 0.2, 0.4, 0.2, 0.2, 0.1],   # bad
                                [0.1, 0.2, 0.2, 0.4, 0.2, 0.1, 0.4, 0.4, 0.3, 0.4, 0.5, 0.2, 0.4, 0.4, 0.2, 0.3],   # good
                                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.3, 0.5, 0.1, 0.3, 0.5, 0.5]],  # very good
                                evidence=[OR, Speed], evidence_card=[4, 4])

cpd_DR = TabularCPD(DR, 4,     [[0.5, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],   # very bad
                                [0.3, 0.2, 0.4, 0.4, 0.2, 0.5, 0.4, 0.3, 0.4, 0.4, 0.1, 0.2, 0.4, 0.2, 0.2, 0.1],   # bad
                                [0.1, 0.2, 0.2, 0.4, 0.2, 0.1, 0.4, 0.4, 0.3, 0.4, 0.5, 0.2, 0.4, 0.4, 0.2, 0.3],   # good
                                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.3, 0.5, 0.1, 0.3, 0.5, 0.5]],  # very good
                                evidence=[def_atk, sp_def_atk], evidence_card=[4, 4])

cpd_DA = TabularCPD(DA, 4,     [[0.5, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],   # very bad
                                [0.3, 0.2, 0.4, 0.4, 0.2, 0.5, 0.4, 0.3, 0.4, 0.4, 0.1, 0.2, 0.4, 0.2, 0.2, 0.1],   # bad
                                [0.1, 0.2, 0.2, 0.4, 0.2, 0.1, 0.4, 0.4, 0.3, 0.4, 0.5, 0.2, 0.4, 0.4, 0.2, 0.3],   # good
                                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.3, 0.5, 0.1, 0.3, 0.5, 0.5]],  # very good
                                evidence=[DR, HP], evidence_card=[4, 4])

cpd_winner = TabularCPD(Winner, 2,
        #--                                                           -                                                           +                                                           ++
        #--              -              +              ++             --             -              +              ++             --             -              +              ++             --             -              +              ++
        # 0.5  1    2    0.5  1    2    0.5  1    2    0.5  1    2    0.5  1    2    0.5  1    2    0.5  1    2    0.5  1    2    0.5  1    2    0.5  1    2    0.5  1    2    0.5  1    2    0.5  1    2    0.5  1    2    0.5  1    2    0.5  1    2
        [[0.1, 0.2, 0.4, 0.1, 0.3, 0.4, 0.2, 0.4, 0.5, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.1, 0.4, 0.6, 0.3, 0.5, 0.7, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.3, 0.5, 0.7, 0.4, 0.6, 0.9, 0.5, 0.7, 0.9, 0.3, 0.5, 0.7, 0.5, 0.6, 0.8, 0.6, 0.7, 0.9, 0.6, 0.8, 0.9],   # pokemon 0 won (self)
         [0.9, 0.8, 0.6, 0.9, 0.7, 0.6, 0.8, 0.6, 0.5, 0.7, 0.5, 0.3, 0.9, 0.7, 0.5, 0.9, 0.6, 0.4, 0.7, 0.5, 0.3, 0.6, 0.4, 0.2, 0.8, 0.6, 0.4, 0.7, 0.5, 0.3, 0.6, 0.4, 0.1, 0.5, 0.3, 0.1, 0.7, 0.5, 0.3, 0.5, 0.4, 0.2, 0.4, 0.3, 0.1, 0.4, 0.2, 0.1]],     # pokemon 1 won (other)
        evidence=[OA, DA, Type], evidence_card=[4, 4, 3])


# G.add_cpds(cpd_atk_def, cpd_sp_atk_def, cpd_def_atk, cpd_sp_def_atk, cpd_HP, cpd_type, cpd_speed, cpd_OR, cpd_DR, cpd_OA, cpd_DA, cpd_winner)
G.add_cpds(cpd_atk_def, cpd_sp_atk_def, cpd_def_atk, cpd_sp_def_atk, cpd_HP, cpd_type, cpd_speed, cpd_OR, cpd_DR, cpd_OA, cpd_DA, cpd_winner)

print(G.check_model())

print("Training")
# Now that we are done with creating the model, we can train the CPD's
prediction_change = True
new_model = None
old_model = G
while prediction_change:
    hidden_estimation = G.predict(df)
    print(hidden_estimation.sample(5))
    prediction_change = False
