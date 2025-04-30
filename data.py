#pip install ucimlrepo

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
steel_plates_faults = fetch_ucirepo(id=198) 
  
# data (as pandas dataframes) 
X = steel_plates_faults.data.features 
y = steel_plates_faults.data.targets 
  
# metadata 
print(steel_plates_faults.metadata) 
  
# variable information 
print(steel_plates_faults.variables) 
