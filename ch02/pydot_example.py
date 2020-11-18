"""
PyDot example from the book.
"""
import pydot as pd

g = pd.Dot(graph_type='graph')

g.add_node(pd.Node(str(0), fontcolor='transparent'))
for i in range(5):
    g.add_node(pd.Node(str(i+1)))
    g.add_edge(pd.Edge(str(0),str(i+1)))
    for j in range(5):
        g.add_node(pd.Node(str(j+1)+'-'+str(i+1)))
        g.add_edge(pd.Edge(str(j+1)+'-'+str(i+1),str(j+1)))
g.write_png('graphs/graph.png')
