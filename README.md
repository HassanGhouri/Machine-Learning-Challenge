# Machine-Learning-Challenge
Scratch Random Forest implementation and an instance of a random forest trained on the machine learning challenge dataset.

Tree is already created and stored in "model_parms.py", use pred.py to use tree to make perdictions. 

RandomForst.py and DecisionTree.py contaian code for scratch Decision Tree and Random Forest implementations, cleaned data set used to train tree is also provided. Trees use entropy to determine attributes and splits.

Enter survey answers in following format and model will perdict which food item it is (Pizza, Shawarma, Sushi):
Format: 

Q1: "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)", Numerical, (1-5).
Q2: "Q2: How many ingredients would you expect this food item to contain?", Freeform Text.
Q3: "Q3: In what setting would you expect this food to be served? Please check all that apply", Categorical, (Week day lunch,Week day dinner,Weekend lunch,Weekend dinner,At a party,Late night snack)
Q4: "Q4: How much would you expect to pay for one serving of this food item?", Freeform Text.
Q5: "Q5: What movie do you think of when thinking of this food item?", Freeform Text.
Q6: "Q6: What drink would you pair with this food item?", Freeform Text.
Q7: "Q7: When you think about this food item, who does it remind you of?", Categorical, (Parents,Siblings,Friends,Strangers,Teachers).
Q8: "Q8: How much hot sauce would you add to this food item?", Categorical, (None, A little (mild), A moderate amount (medium), A lot (hot)).
