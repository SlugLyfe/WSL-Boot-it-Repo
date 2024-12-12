[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/aZ7gHng_)

Lab Notebook

30/11/24
Read in the data
Combine the 3 WSL Seasons
Sort by date of match: Oldest to newest
Create match events Data Frame

1/12/24
Bring in and try to adjust expected Threat grid

3/12/24
Continuing to create and fit expected Threat grid to my data
Created expected threat grid including dribbles
Created expected threat grid considering the next 10 moves

4/12/24
Added xT to event data in original file
Imported events df to separate file
Started working on the throw in probability calculator

5/12/24
Worked on chance of scoring when pressuring opponents thrown in
Started working on passes per defensive action count

6/12/24
Created PPDA dictionary for teams
Calculated Possession Matrix

7/12/24
Finalized possession matrix
Worked on pitch visualizations for different xT models 
Visualizations for PPDA and Possession

8/12/24
Calculate backward pass probabilities for each individual team
Calculate average pressure faced (APF) by possession team

9/12/24
Read in additional stats to help model out
Create Linear Regression model to simulate the backward pass probability using regular xT, possession, and APF, avg league position, and goals scored per game.

10/12/24
Find when a ball passed back should be cleared for the team
Evaluate team and player decision making in terms of % and goal impacts!
Visualizations

Use of AI - I found ChatGPT to be very useful with organizing and commenting code. Organizing my code is something which I tend to do after the fact and which takes me a long time, so in that sense, it saved me alot of time. I also used it to improve visuals and provide alternative ways of structuring my code if I found myself getting stuck, in the latter case it was helpful infrequently and not the most often.

Reflection - 
I realized I ended up implementing aspects of many of the different alternative project styles such as simulate/create data and reproduce a research paper, so thats cool. Due to the amount of metrics I had to create for the project, I simplified my model to be a simple linear regression model using sklearn, varifying the strength of the model via the mean squarred error and the R squared values. The limiting factor of this model is definetely the quantity of Data that I had available to train my model, and to create an accurate label. Due to the small datapool, I believe my label matrix was extremly scewed and would limit the effectiveness of the model. Nevertheless I found some constant negative coeffecients seen with APF and Possesion to be potentially interesting, and had fun seing if my model would act any different to players in back pass situations. (Due to the inflated predicitons my model produced because of the model adjusting too heavily skewed labels, it mostly acted very similar to players). If I had more time I would simulate situations where a player cleared it and calculate the value of a synthetic pass back option and identify trends in that direction. I would also make my possesion and pressing data more specific, and I would find a way to get more WSL event data. To conclude, I had lots of fun!


References:

StatsBomb Open-Data, https://github.com/statsbomb

Lorenzo Cascioli, Max Goldsmith, Luca Stradiotti, et al. “Boot It: A Pragmatic Alternative To Build Up Play”. RBFA Knowledge Center (2024).

Soccermatics - Calculating xT(position based), https://soccermatics.readthedocs.io/en/latest/gallery/lesson4/plot_ExpectedThreat.html
