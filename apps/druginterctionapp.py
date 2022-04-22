import streamlit as st
import pandas as pd
import itertools



df = pd.read_csv('models/druginteraction/drugInteractionShow.csv')
df2 = pd.read_csv('models/druginteraction/drugFoodInteractionShow.csv')


def createDF():
    df_find = pd.DataFrame(columns=['DrugA', 'DrugB', 'Interaction'])
    df_foodfind = pd.DataFrame(columns=['Drug', 'FoodInteractions'])

    return df_find, df_foodfind


def createDrugList(string):
    checkList = string.split(',')

    return checkList


def mergeDrugs(checkList):
    ordered_list = itertools.combinations(checkList, 2)
    drug = list(ordered_list)

    return drug


def matchDrugInteractions(drug, df_find, df_foodfind):
    for data in drug:
        a = data[0]
        b = data[1]

        df_interact = df[(df['DrugA'].str.lower() == a.lower()) & (df['DrugB'].str.lower() == b.lower())]
        df_find = df_find.append(df_interact, ignore_index=True)
        df_find.drop_duplicates(inplace=True)

        df_foodinteract = df2[df2['Drug'].str.lower() == a.lower()]
        df_foodfind = df_foodfind.append(df_foodinteract, ignore_index=True)
        df_foodfind.drop_duplicates(inplace=True)

    return df_find, df_foodfind


def pipeline(drugParam):
    df_find, df_foodfind = createDF()
    checkList = createDrugList(drugParam)
    drug = mergeDrugs(checkList)
    df_find, df_foodfind = matchDrugInteractions(drug, df_find, df_foodfind)

    return df_find, df_foodfind


def app():
    st.header("""Drug Interaction Checker""")

    st.text("""Example: Acenocoumarol,Dabigatran,Zidovudine""")

    user_input = st.text_area("Please input drug names", height=90)
    if user_input != '':
        df_find, df_foodfind = pipeline(user_input)

        st.subheader('Drug Interactions:')
        for i in range(len(df_find)):
            st.write(df_find['DrugA'][i], '-', df_find['DrugB'][i])
            st.write('>', df_find['Interaction'][i])

        st.subheader('Food Interactions:')
        for i in range(len(df_foodfind)):
            st.write(df_foodfind['Drug'][i])
            st.write('  ', df_foodfind['FoodInteractions'][i])

