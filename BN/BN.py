import pgmpy as pgm
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

path_data = "../data/BN/"
def create_BN_4():
    df  = pd.read_csv(path_data + "BN_discrete.csv",   sep=",", header=0)

    G = BayesianModel()
    G.add_nodes_from(df.columns)
    G.add_nodes_from([OA, DA, OR, DR])
    edges = [(atk_def, OR), (sp_atk_def, OR), (Speed, OA), (OR, OA), (def_atk, DR), (sp_def_atk, DR), (HP, DA), (DR, DA), (Type, Winner), (OA, Winner), (DA, Winner)]
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    return G

def create_BN_4():
    df  = pd.read_csv(path_data + "BN_discrete.csv",   sep=",", header=0)

    G = BayesianModel()
    G.add_nodes_from(df.columns)
    G.add_nodes_from([OA, DA])
    edges = [(atk_def, OA), (sp_atk_def, OA), (Speed, OA), (def_atk, DA), (sp_def_atk, DA), (HP, DA), (Type, Winner), (OA, Winner), (DA, Winner)]
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    return G

def initialize_G_4(G):
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

    # print(G.check_model())
    return G


