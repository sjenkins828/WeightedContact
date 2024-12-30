
Each variable of main controls the qualities of the graph, and should be self evident with exception of the first 4. n is the number of nodes. 
The paper used 128, which I have verified my own code on but I have written it to work for any value 2^n. w is the maximum number of connections allowed between two nodes of the multigraph. 
The paper varies this value, with 3 being the average value. m and t are variables related to the power graph specifically. m controls the number of starting connecting before it begins to branch. 
t is the chance to create a cycle, which the paper uses various values for, 50% being one of them.

Mutation events is the main control for how long execution will take. In the repository its 2500 which for a ring graph takes about 30 minutes to run on my machine. 
I'd personally recommend setting it to 250 if you just want to confirm the programs function. The code is set up to work for the Ring Graph. 
If you wish for it work for the Power graph, change all calls to ringGraph with powerGraph, with the required variables. You'll also need to alter the following line

        ani = FuncAnimation(fig,update,frames=framesArray[i], fargs=(graphs[i],n),repeat=False,interval=250)

to be

        ani = FuncAnimation(fig,update2,frames=framesArray[i], fargs=(graphs[i]),repeat=False,interval=250)

You could also remove this line entirely and the following ani.save if you don't want the animated gifs, it will still create non-animated line graphs of infections.
