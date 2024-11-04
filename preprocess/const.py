import json

with open("../data/race_final.json","r") as file:
    RACE = json.load(file)

ICD_DICT = {
    'ICD Certain infectious and parasitic diseases': [
        'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9'
    ], 
    'ICD Neoplasms': [
        'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'D0', 'D1', 'D2', 'D3', 'D4'
    ], 
    'ICD Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism': [
        'D5', 'D6', 'D7', 'D8'
    ], 
    'ICD Endocrine, nutritional and metabolic diseases': [
        'E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'
    ], 
    'ICD Mental, Behavioral and Neurodevelopmental disorders': [
        'F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9'
    ], 
    'ICD Diseases of the nervous system': [
        'G0', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9'
    ], 
    'ICD Diseases of the eye and adnexa': [
        'H0', 'H1', 'H2', 'H3', 'H4', 'H5'
    ], 
    'ICD Diseases of the ear and mastoid process': [
        'H6', 'H7', 'H8', 'H9'
    ], 
    'ICD Diseases of the circulatory system': [
        'I0', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9'
    ], 
    'ICD Diseases of the respiratory system': [
        'J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9'
    ], 
    'ICD Diseases of the digestive system': [
        'K0', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9'
    ], 
    'ICD Diseases of the skin and subcutaneous tissue': [
        'L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9'
    ], 
    'ICD Diseases of the musculoskeletal system and connective tissue': [
        'M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
    ], 
    'ICD Diseases of the genitourinary system': [
        'N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9'
    ], 
    'ICD Pregnancy, childbirth and the puerperium': [
        'O0', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9'
    ], 
    'ICD Certain conditions originating in the perinatal period': [
        'P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9'
    ], 
    'ICD Congenital malformations, deformations and chromosomal abnormalities': [
        'Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9'
    ], 
    'ICD Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified': [
        'R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9'
    ], 
    'ICD Injury, poisoning and certain other consequences of external causes': [
        'S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'
    ], 
    'ICD Codes for special purposes': [
        'U0', 'U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8'
    ], 
    'ICD External causes of morbidity': [
        'V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'W0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9'
    ], 
    'ICD Factors influencing health status and contact with health services': [
        'Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9'
    ]
}

# ALLERGIES = {
#     'food_allergies-nuts_and_seeds': [
#         'Almond', 'Cashew Nut', 'Chestnut', 'Flaxseed', 'Hazelnut', 'Macadamia Nut Oil', 'Nut', 'Peanut', 'Peanut Oil', 'Pecan Nut', 'Pine Nut', 'Pistachio Nut', 'Sesame Seed', 'Sunflower Seed', 'Tree Nut', 'Walnut'], 
#     'food_allergies-fruits_and_vegetables': [
#         'Apricot', 'Artichoke', 'Asparagus', 'Avocado', 'Banana', 'Beet', 'Blackberry', 'Blueberry', 'Broccoli', 'Carrot', 'Celery', 'Cherry', 'Citrus And Derivatives', 'Cucumber (Cucumis Sativus)', 'Eggplant', 'Grape', 'Grapefruit', 'Guava', 'Kiwi', 'Lemon', 'Mango', 'Melon', 'Nectarine', 'Orange', 'Orange Juice', 'Peach', 'Pear', 'Peas', 'Pineapple', 'Plum', 'Pomegranate', 'Potato', 'Pumpkin', 'Pyrus Malus Fruit (Apple)', 'Raspberry', 'Raw Vegetable', 'Squash', 'Strawberry', 'Tomato', 'Watermelon', "Cranberry"
#     ], 
#     'food_allergies-dairy_products': [
#         'Cheese', 'Dairy Aid', 'Lactose', 'Milk', 'Milk Products', 'Whey'
#     ], 
#     'food_allergies-shellfish_and_seafood': [
#         'Crab', 'Fish Oil', 'Fish Product Derivatives', 'Fish Protein', 'Mussels', 'Salmon Oil', 'Scallops', 'Shellfish Derived', 'Shrimp', 'Tuna Oil'
#     ], 
#     'food_allergies-other_foods_and_derivatives': [
#         'Bean', 'Beef Containing Products', 'Beef Protein', 'Chicken Derived', 'Chocolate Flavor', 'Cocoa', 'Coconut', 'Coconut Oil', 'Corn', 'Corn Syrup', 'Egg', 'Egg/Poultry', 'Fava Bean', 'Garlic', 'Gluten', 'Green Tea', 'Honey', 'Inverted Sugar', 'Ketchup', 'Lentils', 'Liver Extract', 'Maitake Mushroom', 'Mayonnaise', 'Mouse Protein', 'Mushroom', 'Mustard', 'Oats', 'Olive', 'Pepper', 'Pork/Porcine Product Derivatives', 'Rice', 'Soy', 'Soybean', 'Sugars, Metabolically Active', 'Turkey', 'Wheat', "Black Pepper","Caffeine","Cinnamon"
#     ], 
#     'drug_allergies-antibiotics_and_antimicrobials': [
#         'Aminoglycosides', 'Amoxicillin', 'Amoxil', 'Ampicillin', 'Augmentin', 'Avelox', 'Azithromycin', 'Bacitracin', 'Bactrim', 'Biaxin', 'Ceclor', 'Cefaclor', 'Cefazolin', 'Cefdinir', 'Cefepime', 'Cefizox', 'Cefpodoxime', 'Ceftin', 'Ceftriaxone', 'Cefuroxime', 'Cefzil', 'Cephalexin', 'Cephalosporins', 'Cipro', 'Cipro Hc', 'Ciprofloxacin', 'Cleocin', 'Clindamycin', 'Erythrocin', 'Erythromycin Base', 'Flagyl', 'Floxin', 'Gentamicin', 'Ivermectin', 'Keflex', 'Levaquin', 'Levofloxacin', 'Linezolid', 'Macrobid', 'Macrolides', 'Metronidazole', 'Minocycline', 'Monurol', 'Moxifloxacin', 'Neomycin', 'Nitrofurantoin', 'Ofloxacin', 'Oxacillin', 'Penicillamine', 'Penicillin G', 'Penicillin V', 'Penicillins', 'Quinolones', 'Rifaximin', 'Rocephin', 'Septra', 'Streptomycin', 'Sulfa (Sulfonamide Antibiotics)', 'Sulfadiazine', 'Sulfalene', 'Sulfamethazine', 'Sulfamethizole', 'Sulfamethoprim', 'Sulfamethoxazole', 'Sulfamethoxazole-Trimethoprim', 'Sulfanilamide', 'Sulfatrim', 'Suprax', 'Teicoplanin', 'Tetracycline', 'Tetracyclines', 'Tobramycin', 'Trimethoprim', 'Trimox', 'Unasyn', 'Vancomycin', 'Zithromax', 'Zithromax Z-Pak', 'Zosyn', "Doxycycline","Mupirocin","Valtrex","Diflucan","Fluconazole","Omnicef","Ribavirin","Terbinafine","Remdesivir (Investigational Use)","Neosporin","Nystatin","Clotrimazole","Tequin","Tamiflu"
#     ], 
#     'drug_allergies-pain_relievers_anti_inflammatory': [
#         'Acetaminophen', 'Advil', 'Aleve', 'Aspirin', 'Bayer Aspirin', 'Capital With Codeine', 'Celebrex', 'Codeine', 'Demerol', 'Diclofenac', 'Dilaudid', 'Dolobid', 'Excedrin Migraine', 'Excedrin Pm', 'Excedrin Sinus Headache', 'Fentanyl', 'Hydrocodone', 'Hydromorphone', 'Ibuprofen', 'Ketorolac', 'Meclomen', 'Meloxicam', 'Mobic', 'Morphine', 'Motrin', 'Motrin Ib', 'Nsaids', 'Naproxen', 'Norco', 'Opioid Analgesics', 'Oxycontin', 'Oxycodone', 'Percocet', 'Suboxone', 'Toradol', 'Tramadol', 'Tylenol', 'Tylenol-Codeine', 'Tylenol-Codeine #3', 'Ultram', 'Vicodin', 'Voltaren', 'Zorvolex',
#         "Alka-Seltzer","Flexeril","Cyclobenzaprine","Robaxin","Imodium A-D","Colchicine","Naloxone","Narcan"
#     ], 
#     'drug_allergies-cardiovascular_drugs': [
#         'Ace Inhibitors', 'Accupril', 'Amlodipine', 'Angiotensin Receptor Antagonist', 'Atenolol', 'Benicar Hct', 'Benazepril', 'Benicar', 'Bisoprolol', 'Brilinta', 'Candesartan', 'Cardizem', 'Carvedilol', 'Clopidogrel', 'Diltiazem', 'Diovan', 'Enalapril', 'Entresto', 'Fosinopril', 'Hytrin', 'Hydralazine', 'Hyzaar', 'Inderal La', 'Irbesartan', 'Labetalol', 'Lisinopril', 'Losartan', 'Metoprolol', 'Nifedipine', 'Nitroglycerin', 'Norvasc', 'Plavix', 'Prasugrel', 'Procardia', 'Propranolol', 'Quinapril', 'Ramipril', 'Sular', 'Telmisartan', 'Terazosin', 'Ticlopidine', 'Valsartan', 'Vasotec', 'Verapamil', 'Zestril',"Altace","Atorvastatin","Crestor","Flomax","Lipitor","Rosuvastatin","Simvastatin","Tamsulosin","Gemfibrozil","Colestipol","Lasix"
#     ], 
#     'drug_allergies-antidepressants_antipsychotics_anxiolytics': [
#         'Abilify', 'Amitriptyline', 'Ativan', 'Buspar', 'Bupropion', 'Caplyta', 'Clonazepam', 'Clozapine', 'Cymbalta', 'Doxepin', 'Duloxetine', 'Escitalopram', 'Fluoxetine', 'Haldol', 'Haloperidol', 'Lexapro', 'Librium', 'Loxapine', 'Mellaril', 'Mirtazapine', 'Olanzapine', 'Paxil', 'Remeron', 'Risperidone', 'Seroquel', 'Serentil', 'Sertraline', 'Tetracyclic Antidepressants', 'Thorazine', 'Trazodone', 'Valium', 'Wellbutrin', 'Xanax', 'Ziprasidone', 'Zyban', 'Zyprexa', "Ambien","Benzodiazepines","Chlorpromazine","Compazine","Lorazepam","Lithium","Adderall","Ritalin"
#     ], 
#     'drug_allergies-respiratory_medications': [
#         'Advair Diskus', 'Albuterol', 'Allegra', 'Azelastine', 'Benadryl', 'Benadryl Allergy', 'Brompheniramine', 'Claritin', 'Clenbuterol', 'Dayquil Sinus Pressure/Pain', 'Dextromethorphan', 'Diphenhydramine', 'Dulera', 'Ephedrine', 'Flovent Hfa', 'Flonase', 'Fluticasone', 'Guaifenesin', 'Hydroxyzine', 'Ipratropium', 'Montelukast', 'Mucinex', 'Nasonex', 'Proair Hfa', 'Pseudoephedrine', 'Robitussin', 'Robitussin A-C', 'Singulair', 'Sudafed', 'Symbicort', 'Theraflu Sinus & Cold', 'Triaminic Cough/Runny Nose', 'Vicks Dayquil', 'Vicks Vaporub', 'Zyrtec', "Atrovent","Benzo-Creme","Benzocaine","Benzonatate","Coricidin","Epinephrine","Phenylephrine","Promethazine"
#     ], 
#     "drug_allergies-gastrointestinal_medications": [
#         "Lansoprazole",
#         "Omeprazole",
#         "Pantoprazole",
#         "Nexium",
#         "Pepcid",
#         "Ranitidine",
#         "Reglan",
#         "Zofran",
#         "Miralax",
#         "Senna",
#         "Pepto-Bismol",
#         "Famotidine",
#         "Gas-X",
#         "Prevpac"
#     ],
#     "drug_allergies-vaccines": [
#         "Covid-19 Vaccine, Mrna, Bnt162B2, Lnp-S (Pfizer)",
#         "Covid-19 Vaccine, Mrna-1273, Lnp-S (Moderna)",
#         "Anthrax Vaccine",
#         "Bcg Vaccine",
#         "Hepatitis B Virus Vaccine",
#         "Influenza Virus Vaccines",
#         "Pneumococcal Vaccine",
#         "Tdvax",
#         "Tetanus Toxoid",
#         "Flucelvax"
#     ],
#     "drug_allergies-local_anesthetics": [
#         "Lidocaine",
#         "Novocain",
#         "Procaine",
#         "Nitrous Oxide"
#     ],
#     'drug_allergies-anticoagulants_antiplatelet': [
#         'Coumadin', 'Eliquis', 'Heparin', 'Lovenox', 'Warfarin',"Pentoxifylline","Truvada"
#     ], 
#     'drug_allergies-anticonvulsants': [
#         'Carbamazepine', 'Depacon', 'Depakote', 'Dilantin', 'Gabapentin', 'Keppra', 'Lamictal', 'Lamotrigine', 'Lyrica', 'Neurontin', 'Oxcarbazepine', 'Phenobarbital', 'Pregabalin', 'Primidone', 'Tegretol', 'Topiramate', 'Valproic Acid', "Baclofen","Clonidine","Guanfacine"
#     ], 
#     'drug_allergies-endocrine_metabolic': [
#         'Alendronate Sodium', 'Anastrozole', 'Arimidex', 'Boniva', 'Creon', 'Cyproterone', 'Finasteride', 'Fosamax', 'Glimepiride', 'Glipizide', 'Januvia', 'Jardiance', 'Kerendia', 'Lantus', 'Levothroid', 'Levothyroxine Sodium', 'Metformin', 'Methimazole', 'Micronase', 'Novolog', 'Rosiglitazone', 'Spironolactone', 'Sulfonylureas', 'Trulicity', 'Victoza',
#         "Levemir","Clomid","Lupron","Iletin Ii Regular(Pork)Conc","Glucose"
#     ], 
#     'drug_allergies-immunosuppressants_chemotherapeutics': [
#         'Arava', 'Avonex', 'Methotrexate', 'Ocrevus', 'Paclitaxel', 'Remicade', "Sulfasalazine","Cortisone","Dexamethasone","Hydrocortisone","Prednisolone","Prednisone","Cortizone-10"
#     ],
#     "drug_allergies-topical_medications": [
#         "Selsun Blue",
#         "Peroxyl",
#         "Itch-X",
#         "Lamisil",
#         "Plantarpatch",
#         "Sulfur-8"
#     ],
#     'drug_allergies-vitamins_supplements': [
#         'Ascor', 'Calcium', 'Cyanocobalamin', 'Ferrous Gluconate', 'Ferrous Sulfate', 'Folic Acid', 'Glutamine', 'Megared Plant-Omega-3', 'Niacin', 'Potassium', 'Thiamine', 'Vitamin D2', 'Vitamin E', "Venofer"
#     ], 
#     'environmental_allergies-pollen_dust_mold': [
#         'Birch', 'Grass Pollen', 'House Dust', 'House Dust Mite', 'Mold', 'Oak', 'Pollen Extracts', 'Poison Ivy Extract', 'Ragweed Pollen', 'Tree And Shrub Pollen', 'Weed Pollen',"Mite-D.Pteronyssinus, Std",
# 	    "Bee Pollen",
#     ], 
#     'environmental_allergies-animal_dander': [
#         'Animal Dander', 'Cat Dander', 'Cow Dander', 'Dog Dander', 'Feathers', 'Horse Dander', 'Rabbit Dander', 'Wool'
#     ], 
#     'environmental_allergies-insect_venom': [
#         'Bee Venom Protein (Honey Bee)', 'Insect Venom', 'Mosquito Eliminator', 'Spider Venom', 'Wasp Venom'
#     ], 
#     'chemical_allergies-metals_elements': [
#         'Aluminum', 'Arsenic', 'Chromium', 'Copper', 'Iron', 'Lead', 'Mercury (Elemental)', 'Nickel', 'Silver', 'Zinc', "Silver Sulfadiazine"
#     ], 
#     'chemical_allergies-latex_rubber': [
#         'Latex', 'Latex, Natural Rubber', 'Ppd Black Rubber Mix', "Adhesive Tape", "Tegaderm"
#     ], 
#     'chemical_allergies-dyes_perfumes': [
#         'Blue Dye', 'Perfume', 'Red Dye'
#     ], 
#     'chemical_allergies-iodine_sulfates': [
#         'Iodine', 'Iodine Containing', 'Ivp Dye, Iodine Containing', 'Sulfamide', 'Sulfite', 'Sulfate Ion', 'Sulfur Dioxide', "Betadine","Gadobutrol","Gadolinium-Containing Agents","Sulfazine","Sulfoam","Thiazides","Hydrochlorothiazide"
#     ]
# }

with open("../data/vacc_final.json", "r") as file:
    VACCINATION = json.load(file)
    
INSURANCE_CLAIM_TYPE = [
    'Central Certification',
    'Other Non-Federal Programs',
    'Preferred Provider Organization (PPO)',
    'Point of Service (POS)',
    'Exclusive Provider Organization (EPO)',
    'Indemnity Insurance',
    'Health Maintenance Organization (HMO) Medicare Risk',
    'Blue Cross/Blue Shield',
    'Champus',
    'Commercial Insurance Co.',
    'Federal Employees Program',
    'Health Maintenance Organization',
    'Liability Medical',
    'Medicare Part A',
    'Medicare Part B',
    'Medicaid',
    'Veterans Affairs Plan',
    'Mutually Defined',
    'Unknown claim type',
    "No Insurance"
]