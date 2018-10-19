# Clustering
This is a side project where I generate random data and cluster them
according to their position in a graph. 

## Inspiration
This project was inspired by the idea of a [4X game](https://en.wikipedia.org/wiki/4X)
(something like [Sid Meier's Civilization V](https://store.steampowered.com/app/8930/Sid_Meiers_Civilization_V/) game) having an AI that
attacked an opposing faction (**F**) based on their distribution of units. Assuming the AI knew the positions of each of **F**'s
units (which is rather unlikely), the AI would determine the front that each unit was on. In this case, it could use the k-means clustering
algorithm, setting k as the number of factions **F** was at war against. Then it would determine if **F**'s units were in an optimal
position and distribution for the AI to attack **F**. 

However, this is not an optimal player experience as we assume that the AI knows the position of every unit of **F**, which is generally
frowned upon in 4X gaming communities. We can remedy this by having the AI save the positions of each unit it sees of **F** and generating
an uncertainty plot for each of those groupings of units. This uncertainty plot would change with each turn, as the AI would determine the
possible new locations for **F**'s units based on their known movement speeds and going in every direction that is possible. Of course,
this uncertainty plot could very easily mean nothing after a small number of turns, but this would theoretically solve the problem of a
dissatisfied fanbase as we no longer assume the AI knows all of the unit positions of **F**.  

## Requirements
The project was built using packages installed from 
[Anaconda 4.5.11](https://www.anaconda.com/download). The 
packages that were used can be found in `requirements.txt`.

## To-do
* Add uncertainty matrix and plot that
  * This needs to be per cluster.
