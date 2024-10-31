from const import SCALE_COLS,SCALE_VAL,SCALE_STD

dict_key = {}
for c,m,s in zip(SCALE_COLS,SCALE_VAL,SCALE_STD):
    dict_key[c] = {
        "mean":m,
        "std":s
    }
    
print(dict_key)