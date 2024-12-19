import numpy as np;
import matplotlib.pyplot as plt;
import networkx as nx;
import random
import datetime
from matplotlib.animation import FuncAnimation;
from multiprocessing import Pool


#creates a ring graph of size n, with degree 4 and the desired shape.
def ringGraph(n):
    G =  nx.MultiGraph()
    for i in range(0,n-1):
        if (i < n//2-1):
            G.add_edge(i,i+1)
        elif (i == n//2 -1):
            G.add_edge(i,0)
        elif (i > n//2 -1 and i !=n-1):
            G.add_edge(i,i+1)
        elif (i == n-1):
            G.add_edge(i, n//2)
    for i in range (n//2, n):
        G.add_edge(i,i-n//2)
        G.add_edge(i,i-n//2+1)
    G.add_edge(0,n-1)
    return G

#create a power graph of size n, creating m connections 
def powerGraph(n, m, t):
    global G
    G = nx.Graph()
    G.add_node(0)
    for node in range(1,n):
        nodes = list(G.nodes)
        if(len(nodes)<m):
            currentNodes = nodes
        else:
            currentNodes = random.sample(nodes,m) #numpy choice wasn't working idk
        for nextNode in currentNodes:
            G.add_edge(nextNode,node)
            if(len(nodes)>m):
                if (np.random.random() < t):
                    neighbors = list(G.neighbors(nextNode))
                    nodeChoice = True
                    triangleNode = 0
                    while(nodeChoice):
                        triangleNode = np.random.choice(neighbors)
                        if(triangleNode == node or triangleNode == nextNode):
                            triangleNode = np.random.choice(neighbors)
                        else:
                            nodeChoice = False
                    G.add_edge(node,triangleNode)
    return G


#draws the ring graph, consisting of two concentric circles. Used with the animation for a graph
def update(frame,G,n):
    outerRadius = 10
    innerRadius = 8
    outerNodes = range(n//2)
    innerNodes = range(n//2,n)
    polarAngle = 2*np.pi/(n//2)
    pos = {}
    for i, node in enumerate(outerNodes):
        pos[node] = (outerRadius*np.cos(i*polarAngle),outerRadius*np.sin(i*polarAngle))
    for i, node in enumerate(innerNodes):
        pos[node] = (innerRadius*np.cos(i*polarAngle),innerRadius*np.sin(i*polarAngle))
    
    colorMap = {0: "#0047AB",1: "#800020", 2: "#818589"}
    nodeColors =[colorMap[value] for value in frame]
    global timeStep
    title = (f"Time Step {timeStep}")
    nx.draw(G,pos=pos,node_color=nodeColors,node_size = 200, edge_color='black' ,ax = ax, with_labels = True)
    ax.set_title(title)
    timeStep = timeStep +1
    
#draws the power graph. Used with the animation for a graph
def update2(frame,G):
    global nodePositions
    if nodePositions is None:
        nodePositions = nx.spring_layout(G)
    
    colorMap = {0: "#0047AB",1: "#800020", 2: "#818589"}
    nodeColors =[colorMap[value] for value in frame]
    nx.draw(G,pos=nodePositions,node_color=nodeColors,node_size = 200, edge_color='black' ,ax = ax, with_labels = True) 

#draw a single unaimated graph based on the passed node embddings
def drawRingGraph(frame,G,n):
    outerRadius = 10
    innerRadius = 8
    outerNodes = range(n//2)
    innerNodes = range(n//2,n)
    polarAngle = 2*np.pi/(n//2)
    pos = {}
    for i, node in enumerate(outerNodes):
        pos[node] = (outerRadius*np.cos(i*polarAngle),outerRadius*np.sin(i*polarAngle))
    for i, node in enumerate(innerNodes):
        pos[node] = (innerRadius*np.cos(i*polarAngle),innerRadius*np.sin(i*polarAngle))
    
    colorMap = {0: "#0047AB",1: "#800020", 2: "#818589"}
    nodeColors =[colorMap[value] for value in frame]
    nx.draw(G,pos=pos,node_color=nodeColors,node_size = 200, edge_color='black' ,ax = ax, with_labels = True)
    global timeStep
    title = (f"Time Step {timeStep}")
    ax.set_title(title)



#adds a connection between two nodes if the maximium number of connections w has not been reached
def add(node1,node2):
    connectivity = G.number_of_edges(node1,node2)
    if (connectivity < w):
        G.add_edge(node1,node2)

#delete a connected if the number of connectionms is greater than zero
def delete(node1,node2):
    if(G.number_of_edges(node1,node2) > 0):
        G.remove_edge(node1,node2)

#add or delete a connected based on the number of connections between two nodes
#NOTE there is suppose to be a boolean value determining the result of the 3rd and 4th options on this list, however the paper does not clarify how to calculate this boolean value
def toggle(node1,node2):
    if(G.number_of_edges(node1,node2) == 0):
        add(node1,node2)
    elif(G.number_of_edges(node1,node2) == w):
        delete(node1,node2)
    elif(node2%2 == 0):
        add(node1,node2)
    else:
        delete(node1,node2)
#if two pairs exist in the graph, than switch the connection between those pairs.
def swap(node1,node2,node3,node4):
    if(node2 in G.neighbors(node1) and node4 in G.neighbors(node3)):
        if (node3 not in G.neighbors(node1) and
            node4 not in G.neighbors(node1) and
            node3 not in G.neighbors(node2) and
            node4 not in G.neighbors(node2)):
            delete(node1, node2)
            delete(node3,node4)
            add(node1,node4)
            add(node2,node3)
#if two pairs exist in the graph, and they share a node, and they dont form a cycle, then change the order of the connections
def hop(node1,node2,node3):
    if(node2 in G.neighbors(node1) and node3 in G.neighbors(node2) and node3 not in G.neighbors(node1)):
        delete(node1,node2)
        add(node1,node3)

#if two pairs exist in the graph, and they share a node, add a connection to make them a cycle
def localAdd(node1,node2,node3):
    if(node2 in G.neighbors(node1) and node3 in G.neighbors(node2)):
        add(node1,node3)

#if two pairs exist in the graph, and they share a node, remove the connect that would form a cycle between them
def localDelete(node1,node2,node3):
    if(node2 in G.neighbors(node1) and node3 in G.neighbors(node2)):
        delete(node1,node3)

#if two pairs exist in the graph, and they share a node, toggle the connect that would form a cycle between them
def localToggle(node1,node2,node3):
    if(node2 in G.neighbors(node1) and node3 in G.neighbors(node2)):
        toggle(node1,node3)

#defines the chromosomes for infection, where each value is 
def defineChromosomes(nChromosomes,chromosomeSize,probabilities):
    chromosomes = []
    for i in range(0,len(probabilities)):
        chromosomes.append([])
        for j in range(0,nChromosomes):
            chromosome = np.random.choice(range(0,len(probabilities[i])),chromosomeSize,p=probabilities[i])
            chromosomes[i].append(chromosome)
    return chromosomes

#modifies the ring graph and returns the result
def performGraphModifcation(chromosome):
    global G
    nodes = list(G.nodes)
    for i in chromosome:
        match i:
            case 0:
                node1,node2 = np.random.choice(nodes,size=2,replace=False)
                toggle(node1,node2)
            case 1:
                node1,node2,node3 = np.random.choice(nodes,size=3,replace=False)
                hop(node1,node2,node3)
            case 2:
                node1,node2 = np.random.choice(nodes,size=2,replace=False)
                add(node1,node2)
            case 3:
                node1,node2 = np.random.choice(nodes,size=2,replace=False)
                delete(node1,node2)
            case 4:
                node1,node2,node3,node4 = np.random.choice(nodes,size=4,replace=False)
                swap(node1,node2,node3,node4)
            case 5:
                node1,node2,node3 = np.random.choice(nodes,size=3,replace=False)
                localToggle(node1,node2,node3)
            case 6:
                node1,node2,node3 = np.random.choice(nodes,size=3,replace=False)
                localAdd(node1,node2,node3)
            case 7:
                node1,node2,node3 = np.random.choice(nodes,size=3,replace=False)
                localDelete(node1,node2,node3)
            case 8:
                pass
    modifiedGraph = G.copy()
    return modifiedGraph
#fitness function for the genetic algorithm. Type 0 is epidemic length, Type 1 would be profile matching, but that has not been implemented
def fitnessFunction(type,results):
    if(type == 0):
        lengths = np.array([len(graph) for graph in results])
        bestIndicies = np.argsort(lengths)[::-1][:2]
        worstIndicies = np.argsort(lengths)[::-1][-2:]
    return bestIndicies, worstIndicies

#performs torunament selection of the two longest running epidemics
def tournamentSelection(size,chromosomes,mutationAmount,probabilities,n,experimentNum):
    chosen = np.random.choice(len(chromosomes),size=size,replace=False)
    modifiedGraphList = []
    for chromosome in range(0,len(chosen)):
        modifiedGraph = performGraphModifcation(chromosomes[chosen[chromosome]])
        modifiedGraphList.append(modifiedGraph)
    framesArray,infectedNumberArray = runAllInfections(modifiedGraphList,n)
    bestIndicies, worstIndicies = fitnessFunction(0,infectedNumberArray)
    offspring1, offspring2 = twoPointCrossover(chromosomes[bestIndicies[0]],chromosomes[bestIndicies[1]])
    offspring1 = mutation(offspring1,mutationAmount,probabilities,experimentNum)
    offspring2 = mutation(offspring2,mutationAmount,probabilities,experimentNum)

    leastFit = []
    for i in worstIndicies:
        leastFit.append(chosen[i])
    
    chromosomes[leastFit[0]] = offspring1
    chromosomes[leastFit[1]] = offspring2
    return chromosomes

#performs two point crossover,in which the parents are severed in two places and reformed to make their children
def twoPointCrossover(parent1, parent2):
    point1 = np.random.randint(1,(len(parent1)-1))
    point2 = np.random.randint(point1+1,parent1.size)

    offspring1 = np.concatenate((parent1[:point1],parent2[point1:point2],parent1[point2:]))
    offspring2 = np.concatenate((parent2[:point1],parent1[point1:point2],parent2[point2:]))
    return offspring1,offspring2

#performs mutations on the given chromosome, up to the amount using the same probabilities
def mutation (chromosome,amount,probabilities,experimentNum):
    amount = np.random.randint(1,(amount+1))
    for i in range(amount):
        mutationPoint = np.random.randint(0,len(chromosome))
        newOperation = np.random.choice(range(0,len(probabilities[experimentNum])),p=probabilities[experimentNum])
        chromosome[mutationPoint] = newOperation
    return chromosome


#run all infections over the given modifed graph list
def runAllInfections(modifiedGraphList,n):
    framesArray = []
    infectedNumberArray = []
    for i in modifiedGraphList:
        nodeEmbeddings = np.zeros(n)
        patientZero = np.random.randint(0,n)
        nodeEmbeddings[patientZero] = 1
        frames,infectedAtTimeStep = runSimpleInfection(nodeEmbeddings,i)
        framesArray.append(frames)
        infectedNumberArray.append(infectedAtTimeStep)
    return framesArray,infectedNumberArray
    

#performs the infection using the SIR model. 0 = Susceptible, 1= Infected, 2= Removed
def runSimpleInfection(nodeEmbeddings,G):
    alpha = 0.5
    totalInfected = 1
    infectedAtTimeStep = []
    frames = []
    
    #get an initial picture of graph before running
    frames.append(nodeEmbeddings.copy())
    while (totalInfected > 0):    #while people are infected
        for i in range(0,len(nodeEmbeddings)): #check every node in the graph
            if nodeEmbeddings[i] == 1: #if the node is infected
                neighborsArray = np.array(list(G.neighbors(i))) 
                for neighbour in neighborsArray: #check each neighbour of the infected indiviual
                    edgeCount = G.number_of_edges(i,neighbour)
                    if nodeEmbeddings[neighbour] == 0: #if the neighbour is susceptible 
                        for _ in range(edgeCount): #try and infect them a number of times equal to the strength of their connection
                            if np.random.random() < alpha:
                                if nodeEmbeddings[neighbour] == 0: #sanity check that is required for some unknown reason
                                    nodeEmbeddings[neighbour] = 1
                                    totalInfected+= 1 
                nodeEmbeddings[i] = 2
                totalInfected -= 1
        frames.append(nodeEmbeddings.copy())
        infectedAtTimeStep.append(totalInfected)
    return frames,infectedAtTimeStep

def main():
    global ax,G,timeStep,w
    timeStep = 0
    n = 128
    w = 3
    m = 1
    t = 0.5
    global nodePositions 
    nodePositions =None
    finalAverage = 50
    nChromosomes = 1000
    chromosomeSize = 256
    tournySize = 7
    mutationAmount = 3
    breedingEvents = 2500
    probabilities = [[0.0438,0.0189,0.0281,0.0499,0.0003,0.4322,0.0051,0.0170,0.4047],
                     [0.0038,0.0156,0.0221,0.0021,0.0285,0.0267,0.4183,0.0396,0.4433],
                     [0.0342,0.0192,0.0044,0.2095,0.0586,0.3327,0.3233,0.0047,0.0134],
                     [0.0176,0.3765,0.0125,0.0157,0.3637,0.0094,0.1757,0.0099,0.0190],
                     [0.3571,0.2016,0.0045,0.0079,0.0026,0.0088,0.3556,0.0510,0.0109],
                     [0.0122,0.0196,0.3839,0.0942,0.0070,0.0067,0.0303,0.0173,0.4288],
                     [0.0038,0.0457,0.4066,0.0769,0.0226,0.0061,0.3951,0.0268,0.0164],
                     [0.0020,0.0727,0.0010,0.0435,0.0355,0.0123,0.7945,0.0201,0.0184]]

    startTime = datetime.datetime.now()
    #create the population of candidate chromosomes.   
    chromosomes = defineChromosomes(nChromosomes,chromosomeSize,probabilities)
    
    #begin an experiment
    nodeEmbeddings = np.zeros(n)
    G = ringGraph(n)
    chosen = []
    #perform all experiments on the graph, selecting 50 from the final infection
    for chromosome in range(len(chromosomes)):
        chosen.append([])
        print(f"Start Experiment {chromosome+1}")
        for breedingEvent in range(breedingEvents):
            chromosomes[chromosome] = tournamentSelection(tournySize,chromosomes[chromosome],mutationAmount,probabilities,n,chromosome)
        choices = np.random.choice(range(len(chromosomes[chromosome])),size=finalAverage,replace=False)
        for choice in choices:
            chosen[chromosome].append(chromosomes[chromosome][choice])

    #rerun the infection on the selected infection, average their results
    runningTotals = np.zeros(len(chosen))
    framesArray =[]
    frameChoice = np.random.randint(0,(finalAverage),size=len(chosen))
    graphs = []
    resultsArray = []
    for i in range(len(chosen)):
        for j in range(len(chosen[i])):
            G = ringGraph(n)
            graph = performGraphModifcation(chosen[i][j])
            nodeEmbeddings = np.zeros(n)
            patientZero = np.random.randint(0,n)
            nodeEmbeddings[patientZero] = 1
            frames,results = runSimpleInfection(nodeEmbeddings,graph)
            runningTotals[i] += len(results)
            if(frameChoice[i]==j):
                framesArray.append(frames)
                graphs.append(graph)
                resultsArray.append(results)
        runningTotals[i] /= finalAverage
        print(f"Done Experiment {i+1}")
    doneExperiments = datetime.datetime.now()
    print(f"Execution time for experiments {doneExperiments-startTime}")
    print(f"Averages {runningTotals}")
    #stop 
    for i in range(len(framesArray)):
        fig, ax = plt.subplots()
        ani = FuncAnimation(fig,update,frames=framesArray[i], fargs=(graphs[i],n),repeat=False,interval=250)
        ani.save((f"infectionSimulation{i+1}.gif"),writer="pillow",fps=2)
        ax.clear()
        fig.clear()
        timeStep = 0
        plt.plot(resultsArray[i])
        plt.title(f"Experiment {i+1} Total Infected Over Time")
        plt.xlabel("Time Steps")
        plt.ylabel("Total Infected")
        plt.grid(True)
        plt.savefig(f"totalInfectedPlot{i+1}.png")
        ax.clear()
        fig.clear()
        print(f"Done Animation {i+1}")
    print(f"Execution time for animations {datetime.datetime.now()-doneExperiments}")
main()