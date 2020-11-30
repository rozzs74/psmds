import numpy as np
import matplotlib.pyplot as plt

from pandas import read_csv
from pandas import DataFrame
from typing import List

# Step 1 Define Data set
outlook: List[str] = ["sunny", "sunny", "overcast", "rain", "rain", "rain", "overcast", "sunny", "sunny", "rain", "sunny", "overcast", "overcast", "rain"]
temperature: List[str] = ["hot", "hot", "hot", "mild", "cool", "cool", "cool","mild", "cool", "mild", "mild", "mild", "hot", "mild"]
humidity: List[str]= ["high", "high", "high", "high", "normal", "normal", "normal", "high", "normal", "normal", "normal", "high", "normal", "high"]
windy: List[str] = ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"]
decision: List[str] = ["N", "N", "P", "P", "P", "N", "P", "N", "P", "P","P", "P", "P", "N" ]
columns: List[str]=["Outlook", "Temperature", "Humidity", "Windy", "Decision"]

data: dict(Outlook=list, Temperature=list, Humidity=list, Windy=list, Decisions=list) = {
	"Outlook": outlook,
	"Temperature": temperature,
	"Humidity": humidity,
	"Windy": windy,
	"Decisions": decision
}
df: DataFrame = DataFrame(data=data)
df_head: DataFrame = df.head()
df_tail: DataFrame = df.tail()

# Step 2 Compute the Entropy value in this case "decision" column would be the context of information theory
def get_entropy(df) -> float:
	probability_of_p: float = len(df.loc[df.Decisions == "P"]) / len(df) # 9 / 14
	probability_of_n: float = len(df.loc[df.Decisions == "N"]) / len(df) # 5 / 14
	entropy: float = -probability_of_n * np.log2(probability_of_p) - probability_of_p * np.log2(probability_of_p)
	print(f"H(s) = {entropy}")
	return entropy

# Step 3 Compute for the information gain in this case the remaining columns other than decision would be use
# Calculate which attribute provides highest information gain between Outlook, Temperature, Humidity, and Windy
# Starts with Outlook

IG_decision_Outlook: float = get_entropy(df)
IG_equation: str = "IG(Decision, Outlook) = Entropy(Decision)"

for name, Outlook in df.groupby("Outlook"):
	print(name)
	print(Outlook)
	num_p = len(Outlook.loc[Outlook.Decisions == "P"])
	num_n = len(Outlook.loc[Outlook.Decisions == "N"])
	num_Outlook = len(Outlook)
	print(f'p(Decision=P|Outlook={name}) = {num_p}/{num_Outlook}')
	print(f'p(Decision=N|Outlook={name}) = {num_n}/{num_Outlook}')    
	print(f'p(Decision|Outlook={name}) = {num_Outlook}/{len(df)}')
	print(f'Entropy(Decision|Outlook={name}) = '\
         f'-{num_p}/{num_Outlook}.log2({num_p}/{num_Outlook}) - '\
         f'{num_n}/{num_Outlook}.log2({num_n}/{num_Outlook})')

	entropy_decision_outlook = 0
	# Cannot compute log of 0 so add checks
	if num_p != 0:
		entropy_decision_outlook -= (num_p / num_Outlook) \
		* np.log2(num_p / num_Outlook)

	# Cannot compute log of 0 so add checks
	if num_n != 0:
		entropy_decision_outlook -= (num_n / num_Outlook) \
		* np.log2(num_n / num_Outlook)

	IG_decision_Outlook	-= (num_Outlook / len(df)) * entropy_decision_outlook
	IG_equation += f' - p(Decision|Outlook={name}).'
	IG_equation += f'Entropy(Decision|Outlook={name})'
	print(IG_equation)
	print(f'Gain(Decision, Outlook) = {IG_decision_Outlook:0.4f}')

# Step 4 Wrap this in Function IG