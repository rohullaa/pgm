#Implementation of Algorithm A.2: 
#Maximum weight spanning tree in an undirected graph

import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BicScore, HillClimbSearch

import networkx as nx
import matplotlib.pyplot as plt

def get_max_weight(list_visited, weights):
    #Going through every V's and checking whether if it is not visited and it has weight 
    #bigger maximal_weight. If so, then it gets registered as visited and the index is returned.

	maximal_weight = -10_000
	for i in range(V):
		if list_visited[i] == False and weights[i] > maximal_weight:
			maximal_weight = weights[i]
			index_max_weight = i
	return index_max_weight

def Max_Weight_Spanning_Tree(graph):
    list_weights = [-10000 for i in range(V)] #initiliazing all the V's with large negative weight
    list_visited = [False for i in range(V)] #list_visited: wheter a node has been visisted or not. We start by saying that none has been visited.
    list_parent = [0 for i in range(V)] #list for parents of each node

    #We start by a random Node. We assume that it has very high weight and no parent.
    list_weights[0] = 10_000
    list_parent[0] = -1

    for _ in range(V):
        index_max_weight =  get_max_weight(list_visited, list_weights) 
        list_visited[index_max_weight] = True 

        for j in range(V):
            if (graph[j][index_max_weight] != 0 and list_visited[j] == False):
                if (graph[j][index_max_weight] > list_weights[j]):
                    list_weights[j] = graph[j][index_max_weight]
                    list_parent[j] = index_max_weight

    return list_parent, graph

def draw_Graph(list_edges):
    # Defining the model structure. 
    # We can define the network by just passing a list of edges.
    model = BayesianModel(list_edges)
    nx.draw(model, with_labels = True); 
    plt.show()

    return model

def compute_BIC(model, df):
    bic = BicScore(df)

    bic_score = bic.score(model)
    print(f"BIC_score: {bic_score}")
    
    return bic_score

def find_better_DAG():
    #using the HillClimbSearch to find a better DAG
    hc = HillClimbSearch(df)
    best_model = hc.estimate(scoring_method=BicScore(df))
    return best_model.edges()

if __name__ == '__main__':
    #reading the data 
    path = "https://www.uio.no/studier/emner/matnat/math/STK4290/v22/bn_data.csv"
    df = pd.read_csv(path)

    #converting the dataframe into a nested list
    data = df.values.tolist()

    #defining the names of the columns and length of columns
    V_names = ["V1","V2", "V3" ,"V4", "V5" ,"V6","V7", "V8", "V9", "V10", "V11"]
    V = len(data[0])

    #task 4a) Finding the Maximum weight spanning tree and printing the results
    list_parent,data = Max_Weight_Spanning_Tree(data)
    weight_max_tree = sum([data[i][list_parent[i]] for i in range(1,V)])
    print("The aximum Spanning-tree ", weight_max_tree)  
    for i in range(1, V):
        print(f"Edges: {V_names[list_parent[i]]} - {V_names[i]}  Weigth: {data[i][list_parent[i]] } ")


    #making a list of edges and draw the graph found in 4a
    list_edges= [(V_names[list_parent[i]],V_names[i]) for i in range(1,V)]
    model = draw_Graph(list_edges)

    #computing the bic_Score for the graph
    bic_score = compute_BIC(model,df)


    #using HillClimbSearch to find better DAG. Drawing and finding the BIC for his model:
    new_edges = find_better_DAG()
    model = draw_Graph(new_edges)
    bic_score = compute_BIC(model,df)

