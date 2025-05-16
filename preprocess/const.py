import json

with open("./preprocess/race_final.json","r") as file:
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

with open("./preprocess/vacc_final.json", "r") as file:
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

SCALE_COLS = {
    'Age': {'mean': 61.29849357562402, 'std': 18.19418807291505},
    'Pulse': {'mean': 77.08666582476043, 'std': 7.968495718027011}, 
    'Systolic': {'mean': 127.54706114658761, 'std': 10.966879454975894}, 
    'Diastolic': {'mean': 76.12535998482291, 'std': 7.157650023686608},
    'Temperature': {'mean': 97.37746139507836, 'std': 0.4029931929148571}, 
    'BMI': {'mean': 28.31795665485163, 'std': 4.738855199226916}, 
    'RespiratoryRate': {'mean': 15.769906806459147, 'std': 0.8270299872958025}, 
    'OxygenSaturation': {'mean': 97.54472085087278, 'std': 0.7741982235415734}, 
    'OxygenConcentration': {'mean': 21.0, 'std': 0.0}, 
    'CancelledRateThePast6Months': {'mean': 0.14599177522116094, 'std': 0.13431310129158702}, 
    'RescheduledRateThePast6Months': {'mean': 0.15612077044526768, 'std': 0.1339384312562343}, 
    'CancelledAppointmentsSinceLastEncounter': {'mean': 0.2672286817380437, 'std': 0.6500633444773264}, 
    'RescheduledAppointmentsSinceLastEncounter': {'mean': 0.30636702448168895, 'std': 0.7095928735594833}, 
    'Current ICD Count': {'mean': 13.32399741123435, 'std': 10.304246862455331}, 
    '6months ICD Count': {'mean': 21.18070030574215, 'std': 11.426550590906622}, 
    'allergies_count': {'mean': 0.39879265326162155, 'std': 0.9348012162877456}, 
    'vaccination_count': {'mean': 0.24408043027070456, 'std': 0.6698808172313843}, 
    'Average Visit Pattern': {'mean': 36.18208549930163, 'std': 38.270728456473265},
    'Target': {'mean': 39.51687802487288, 'std': 66.45219176648358}, 
    "Target_shift_1":{}
}


ROBUST_SCALE_NO_LOG = {
    "center":[58.32054794520548, 77.08666582476043, 127.5470611465876, 76.12535998482291, 97.37746139507836, 28.31795665485163, 15.769906806459147, 97.54472085087278, 21.0, 0.111111111, 0.125, 0.0, 0.0, 9.0, 15.0, 0.0, 0.0, 29.5, 18.0,21.0],
    "scale":[28.43287671232877, 1.0866658247604306, 1.5470611465875947, 0.12535998482290722, 1.0, 1.117956654851632, 0.23009319354085278, 0.45527914912722167, 1.0, 0.230769231, 0.25, 1.0, 1.0, 14.0, 14.0, 1.0, 1.0, 42.0, 32.0,42.0]
}

ROBUST_SCALE_CONST = {
    "center":[58.32054794520548, 77.08666582476043, 127.5470611465876, 76.12535998482291, 97.37746139507836, 28.31795665485163, 15.769906806459147, 97.54472085087278, 21.0, 0.111111111, 0.125, 0.0, 0.0, 9.0, 15.0, 0.0, 0.0, 3.417726683613366, 2.9444389791664403, 3.091042453358316],
    "scale":[28.43287671232877, 1.0866658247604306, 1.5470611465875947, 0.12535998482290722, 1.0, 1.117956654851632, 0.23009319354085278, 0.45527914912722167, 1.0, 0.230769231, 0.25, 1.0, 1.0, 14.0, 14.0, 1.0, 1.0, 1.203972804325936, 1.6094379124341005, 1.8908503718722862]
}

STANDARD_SCALE_CONST = {
    "mean":[56.29811049012031, 77.13520334460752, 127.14374221136244, 76.17135591045547, 97.39406340566542, 28.199370646609655, 15.77630142959193, 97.55644214566648, 21.0, 0.13906234253752933, 0.14281092871022388, 0.27694255254595485, 0.2885554962702573, 11.248778548486655, 17.378574990671282, 0.33551991462917047, 0.20492486460284082, 3.4072436027155804, 2.9467899121051113],

    "var": [396.13476378593543, 69.38740091037967, 132.74548789560498, 56.364600294247026, 0.17109915481911386, 22.718995846055346, 0.7344734354216104, 0.6135089370826934, 0.0, 0.022078945792841354, 0.02172351739269646, 0.4527519226733766, 0.4851806024506105, 87.54486287549952, 126.37203063764017, 0.6954850026103928, 0.42377900221520526, 1.4779254413266922, 1.5382927243281834],
    
    "scale":[19.90313452162587, 8.329910018144234, 11.521522811486552, 7.507636132248754, 0.41364133596524644, 4.766444780552414, 0.8570142562534245, 0.7832681131532762, 1.0, 0.14858985763786622, 0.14738900024322188, 0.6728684289468311, 0.6965490667932953, 9.356541181200429, 11.241531507656783, 0.833957434531519, 0.650983104400725, 1.21569956869561, 1.2402792928724495]
}

LOG_COLS = [
    "Average Visit Pattern","Target","Target_shift_1"
]

SEED = 46