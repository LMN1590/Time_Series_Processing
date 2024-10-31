from bs4 import BeautifulSoup
from typing_extensions import TypedDict
from typing import Dict
import json


with open("data/vacc.xml","r") as file:
    content = file.read()

vaccinations = {}

raw_xml = BeautifulSoup(content,"xml")
rows = raw_xml.findChildren("CVXVGInfo")
xml_text = [[child.text.strip() for child in row.findChildren("Value") ] for row in rows]
for text in xml_text:
    cur_item = {
        "name":text[0],
        "cvx_code":text[1]
    }
    vaccinations[text[3]] = vaccinations.get(text[3],[]) + [cur_item]
    
with open("data/vacc_final.json","w") as file:
    print(json.dumps(vaccinations),file=file)