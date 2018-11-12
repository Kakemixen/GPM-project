import numpy as np
import pandas as pd

# df = pd.read_csv(data_file_name, sep=',', names=header)
matches_df  = pd.read_csv("data/combats.csv",   sep=",", header=0)
pokemon_df  = pd.read_csv("data/pokemon.csv",   sep=",", header=0, index_col="#")
# pokemon_df.set_index("#")
tests       = pd.read_csv("data/tests.csv",     sep=",", header=0)

# print(matches_df.loc[[0,1]])
# print(pokemon_df.loc[[0,1]])

def get_processed_fight(i):
    """
    @params
        i   :   int
            the row in matches table to return prrocessed

    @return
        match   :   pandas DataFrame
            the match consisting of advantage entries instead of absolute entries

            -1 : disadvantage
            0  : no advantage
            1  : advantage
    """
    pokemon_0_id = matches_df.at[i, 'First_pokemon']
    pokemon_1_id = matches_df.at[i, 'Second_pokemon']
    winner = 1 if matches_df.at[i, 'Winner'] == pokemon_1_id else 0

    pokemon_0 = pokemon_df.loc[[pokemon_0_id]]
    pokemon_1 = pokemon_df.loc[[pokemon_1_id]]

    type_advantage = offensive_type_advantage(pokemon_df.at[pokemon_0_id, "Type 1"], pokemon_df.at[pokemon_1_id, "Type 1"])

    statnames = pokemon_df.columns[3:-2] #HP, Atk, Def, Sp.Atk, Sp.Def, Spd

    number_advantages = pd.DataFrame()
    total_advantage = 0
    for column in statnames:
        name = column + " advantage"
        number_advantages[name] = np.array([numbers_advantage(
            pokemon_df.at[pokemon_0_id, column],
            pokemon_df.at[pokemon_1_id, column])
        ])
        total_advantage += number_advantages[name][0]

    number_advantages["Avg. Advantage"] = np.array([total_advantage / len(statnames)])

    #everything is advantages for pokemon_0
    Atk_Def = pokemon_df.at[pokemon_0_id, "Attack"] / pokemon_df.at[pokemon_1_id, "Defense"]
    Sp_Atk_Def = pokemon_df.at[pokemon_0_id, "Sp. Atk"] / pokemon_df.at[pokemon_1_id, "Sp. Def"]
    Def_Atk = pokemon_df.at[pokemon_0_id, "Defense"] / pokemon_df.at[pokemon_1_id, "Attack"]
    Sp_Def_Atk = pokemon_df.at[pokemon_0_id, "Sp. Def"] / pokemon_df.at[pokemon_1_id, "Sp. Atk"]


    match_advantages = pd.DataFrame(number_advantages)
    match_advantages["Type Advantage"] = np.array([type_advantage])
    match_advantages["Atk/Def ratio"] = np.array([Atk_Def])
    match_advantages["Sp.Atk/Def ratio"] = np.array([Sp_Atk_Def])
    match_advantages["Def/Atk ratio"] = np.array([Def_Atk])
    match_advantages["Sp.Def/Atk ratio"] = np.array([Sp_Def_Atk])
    match_advantages["Winner"] = np.array([winner])

    return match_advantages

def get_processed_data():
    processed_table = get_processed_fight(0)
    for i in range(1, len(matches_df.index)):
        row = get_processed_fight(i)
        processed_table.loc[i] = row.loc[0]

    return processed_table


def get_winner(i):
    return 1 if (matches_df.at[i, 'Winner'] == matches_df.at[i, "Second_pokemon"]) else 0


def numbers_advantage(pokemon_0_stat, pokemon_1_stat):
    return pokemon_0_stat - pokemon_1_stat

def binary_numbers_advantage(pokemon_0_stat, pokemon_1_stat):
    if pokemon_0_stat > pokemon_1_stat:
        return 1
    if pokemon_0_stat < pokemon_1_stat:
        return -1
    return 0

def offensive_type_advantage(type_0, type_1):
    """
    returns the advantage type_0 has attacking type_1
    """

    type_order = ['Normal','Fire','Water','Electric','Grass','Ice','Fighting','Poison','Ground','Flying','Psychic','Bug','Rock','Ghost','Dragon','Dark','Steel','Fairy']
    type_index = dict()
    for i in range(len(type_order)):
        type_index[type_order[i].lower()] = i

    # advantage column attack row
    advantage_matrix = [[1,1,1,1,1,1,2,1,1,1,1,1,1,0,1,1,1,1], \
                        [1,0.5,2,1,0.5,0.5,1,1,2,1,1,0.5,2,1,1,1,0.5,0.5], \
                        [1,0.5,0.5,2,2,0.5,1,1,1,1,1,1,1,1,1,1,0.5,1], \
                        [1,1,1,0.5,1,1,1,1,2,0.5,1,1,1,1,1,1,0.5,1], \
                        [1,2,0.5,0.5,0.5,2,1,2,0.5,2,1,2,1,1,1,1,1,1], \
                        [1,2,1,1,1,0.5,2,1,1,1,1,1,2,1,1,1,2,1], \
                        [1,1,1,1,1,1,1,1,1,2,2,0.5,0.5,1,1,0.5,1,2], \
                        [1,1,1,1,0.5,1,0.5,0.5,2,1,2,0.5,1,1,1,1,1,0.5], \
                        [1,1,2,0,2,2,1,0.5,1,1,1,1,0.5,1,1,1,1,1], \
                        [1,1,1,2,0.5,2,0.5,1,0,1,1,0.5,2,1,1,1,1,1], \
                        [1,1,1,1,1,1,0.5,1,1,1,0.5,2,1,2,1,2,1,1], \
                        [1,2,1,1,0.5,1,0.5,1,0.5,2,1,1,2,1,1,1,1,1], \
                        [0.5,0.5,2,1,2,1,2,0.5,2,0.5,1,1,1,1,1,1,2,1], \
                        [0,1,1,1,1,1,0,0.5,1,1,1,0.5,1,2,1,2,1,1], \
                        [1,0.5,0.5,0.5,0.5,2,1,1,1,1,1,1,1,1,2,1,1,2], \
                        [1,1,1,1,1,1,2,1,1,1,0,2,1,0.5,1,0.5,1,2], \
                        [0.5,2,1,1,0.5,0.5,2,0,2,0.5,0.5,0.5,0.5,1,0.5,1,0.5,0.5], \
                        [1,1,1,1,1,1,0.5,2,1,1,1,0.5,1,1,0,0.5,2,1]]
    return advantage_matrix[type_index[type_1.lower()]][type_index[type_0.lower()]]

if __name__ == "__main__":
    print("starting to process data...")
    data = get_processed_data()
    print("have file, writing to file...")
    data.to_csv("data/advantage_matches.csv", mode="w+")
    print("Done!")
