import json
from dataclasses import dataclass
from typing_extensions import TypedDict
from typing import List
from functools import reduce

with open("../data/race_filtered.json", "r") as file:
    content = json.load(file)

class Concept(TypedDict):
    code:str
    display:str
    child_concept:List["Concept"]

def get_base_concept(child_concepts,level:int = 0,child_concepts_key:str="concept")->List[Concept]:
    if(level!=0):
        base_child=[Concept(
            code = concept["code"],
            display = concept["display"],
            child_concept=get_base_concept(concept[child_concepts_key],level=level-1) if child_concepts_key in concept else []
        ) for concept in child_concepts]
        return base_child
    else: 
        base_child = [Concept(
            code = concept["code"],
            display = concept["display"],
            child_concept=[]
        ) for concept in child_concepts]
        
        far_child = [
            get_base_concept(concept[child_concepts_key]) for concept in child_concepts if child_concepts_key in concept
        ]
        
        far_child = reduce(lambda arr,x:arr+x, far_child, [])
        return base_child+far_child
        
def get_concept_json(concept:Concept)->List[Concept]:
    is_end = any(len(child_concept["child_concept"])==0 for child_concept in concept["child_concept"])
    if(is_end):
        concept = Concept(
            code=concept["code"],
            display=concept["display"],
            child_concept=get_base_concept(concept["child_concept"],level=0,child_concepts_key="child_concept")
        )
        return [concept]
    else:
        finalized_concepts = [get_concept_json(child_concept) for child_concept in concept["child_concept"]]
        return reduce(lambda arr,x:arr+x,finalized_concepts,[])
    
    
    

new_race_list = []
for concept in content:
    new_item = {
        "code": concept["code"],
        "display": concept["display"],
        "child_concept": get_base_concept(concept["concept"],level=3) if "concept" in concept else []
    }
    finalized_json = get_concept_json(new_item)
    new_race_list+=finalized_json


with open("../data/race_final.json","w") as file:
    print(json.dumps(new_race_list),file=file)

# print(content["concept"])