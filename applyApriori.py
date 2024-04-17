import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pickle


def read_items(filename):
    with open(filename,'r', encoding="utf-8") as file:
        lines=file.readlines()
   
    items=dict()
    for line in lines:
        withoutComma=line.split(",")[0]
        key=withoutComma.split(":")[0].strip()
        item=withoutComma.split(":")[1].strip()
        items[key]=item

    return items

        
def read_csvfile(filename):
    data=pd.read_csv(filename, header=None)
    veri=data.copy()
    veri.columns= ["Urun"]
    veri = list(veri["Urun"].apply(lambda x: x.split(",")))
    
    return veri

def save_and_load_pkl(te):
    with open('transaction_encoder.pkl', 'wb') as f:
        pickle.dump(te, f)

    with open('transaction_encoder.pkl', 'rb') as f:
        loaded_te = pickle.load(f)

    return loaded_te

def te_transform(loaded_te, veri):
    custom_te=loaded_te.transform(veri)
    custom_te=pd.DataFrame(custom_te)

    return custom_te

def return_df_apriori(custom_te, min_support, use_colnames, metric, min_threshold):
    df1=apriori(custom_te, min_support, use_colnames)
    df1=association_rules(df1, metric, min_threshold)

    return df1


def test_data(data, df, loaded_te):
    my_filter=df['antecedents'].apply(lambda x:set(data).issubset(set(x)))
    filtered_df=df[my_filter].sort_values(by='support', ascending=False)
    max_support_data= filtered_df.iloc[0]
    consequents=next(iter(max_support_data['consequents']))
    id=loaded_te.columns_[consequents]

    return consequents,id


def main():
    items=read_items('items.txt')
    veri=read_csvfile("veri.csv")
    
    te=TransactionEncoder()
    te.fit(veri)

    loaded_te=save_and_load_pkl(te)
    custom_te=te_transform(loaded_te,veri)

    min_support=0.0001
    use_colnames=True
    metric="confidence"
    min_threshold=0.05
    df=return_df_apriori(custom_te,min_support,use_colnames,metric,min_threshold)
    print(df)

    new_data=[40,41]
    consequent,id=test_data(new_data,df,loaded_te)

    print(f"index= {consequent}, id= {id}, isim= {items.get(id)}\n")
    
    

if __name__=="__main__":
    main()